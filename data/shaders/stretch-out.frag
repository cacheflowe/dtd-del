
// Stretch out from an origin point and rotation.
// The two halves are pushed apart by uAmp.
// The gap can either be filled by stretching a thin band of source pixels (uGapMode=0) or left transparent (uGapMode=1).
// The push can be split evenly (uWeight=0.5) or biased to one side (0=all push to negative side, 1=all push to positive side).

uniform vec2 uOrigin;
uniform float uAmp;
uniform float uBufferWidth;
uniform float uRotation; // radians, angle of the split line's normal
uniform float uWeight; // 0-1, how much of the push goes to the positive-normal side; 0.5 = even split
uniform float uGapMode; // false = stretch the buffer zone to fill the gap; true = leave it transparent

out vec4 fragColor;
void main()
{
	// split along a line through uOrigin (rotated by uRotation) and push both halves apart by uAmp,
	// undistorted beyond the buffer zone; inside it, a thin band of source pixels near the split
	// stretches to fill the gap
	float width = 1.0 / uTDOutputInfo.res.z;
	float height = 1.0 / uTDOutputInfo.res.w;
	vec2 aspect = width * uTDOutputInfo.res.wz; // work in a square-pixel space so rotation looks correct

	// n is the split line's normal direction; d is the signed distance of this pixel from the line
	vec2 n = vec2(cos(uRotation), sin(uRotation));
	vec2 rel = (vUV.st - uOrigin) / aspect;
	float d = dot(rel, n);
	// tangent is the component of rel running along the split line - untouched by the stretch
	vec2 tangent = rel - d * n;

	float bufferWidth = max(uBufferWidth, 1e-5); // avoid divide-by-zero when the buffer is 0
	// each side's own full push amount - uWeight=0.5 splits uAmp*2 evenly (uAmp each);
	// at uWeight=0/1 one of these hits exactly 0, so that side ends up perfectly stationary
	float posAmp = uAmp * 2.0 * uWeight;
	float negAmp = uAmp * 2.0 * (1.0 - uWeight);

	float gapAlpha = 1.0;
	float sampleDistSigned;
	float side = sign(d);
	float ad = abs(d);
	float sideAmp = d >= 0.0 ? posAmp : negAmp;
	if(uGapMode > 0.5) {
		// leave the gap transparent instead of stretching it: sample as if undistorted (only
		// visible right at the antialiased edge) and fade alpha to 0 towards the split line.
		// fwidth() sizes the antialiased edge to a constant ~1px regardless of zoom/resolution
		sampleDistSigned = d - side * sideAmp;
		float aa = fwidth(d) * 1.5;
		gapAlpha = smoothstep(sideAmp - aa, sideAmp + aa, ad);
	} else {
		// stretch a source band of width uBufferWidth to cover this side's screen zone
		// (uBufferWidth + sideAmp). The old pow(t, power) curve put ~zero slope right at the split
		// and dumped all the actual stretching into a thin sliver next to the outer edge, which read
		// as a hard clamp with a late snap rather than a gradual pull. A cubic Hermite spline fixes
		// this: it still matches slope 1 at the outer edge (seamless blend into the untouched image)
		// but starts at the split with the zone's OWN average slope (bufferWidth/stretchedScreenWidth)
		// instead of 0, so the stretch is visibly, gradually happening across the whole band as you
		// move in from the edge, not just snapping at the last moment. Still fully monotonic (no
		// folding), and collapses to a plain line (no distortion) when sideAmp is 0.
		float stretchedScreenWidth = bufferWidth + sideAmp;
		if(ad >= stretchedScreenWidth) {
			// outside the zone entirely: plain 1:1 shift, completely undistorted
			sampleDistSigned = side * (ad - sideAmp);
		} else {
			float t = ad / stretchedScreenWidth;
			float t2 = t * t;
			float t3 = t2 * t;
			float h10 = t3 - 2.0 * t2 + t;   // tangent-at-0 basis
			float h01 = -2.0 * t3 + 3.0 * t2; // value-at-1 basis
			float h11 = t3 - t2;             // tangent-at-1 basis
			// start tangent uses the UNWEIGHTED bufferWidth/(bufferWidth+uAmp) slope (not this
			// side's own sideAmp), so both sides start with the exact same curvature right at the
			// split regardless of uWeight - otherwise the lightly-pushed side's own slope here
			// (bufferWidth/stretchedScreenWidth) approaches 1 as sideAmp shrinks towards 0, going
			// fully flat/undistorted at the seam and creating a visible kink against the heavily-
			// pushed side's much softer curve. Sharing one baseline slope keeps the seam looking
			// equally "worked on" by uBufferWidth on both sides, no matter how uWeight is split
			float m0 = (bufferWidth / (bufferWidth + uAmp)) * stretchedScreenWidth;
			// end tangent = 1 (matches outside), scaled to t-space, but clamped to 3x the secant
			// slope (Fritsch-Carlson bound) so the Hermite curve can't overshoot/fold when sideAmp
			// is very large relative to a small uBufferWidth - trades a tiny slope mismatch at the
			// outer edge for guaranteed monotonicity in that extreme case
			float m1 = min(stretchedScreenWidth, 3.0 * bufferWidth);
			float sampleDist = h10 * m0 + h01 * bufferWidth + h11 * m1;
			sampleDistSigned = side * sampleDist;
		}
	}

	// rebuild the sample position: unchanged tangent + the warped distance back along the normal,
	// then convert back out of square-pixel space into UV space
	vec2 sampleUV = uOrigin + (tangent + sampleDistSigned * n) * aspect;
	vec4 color = texture(sTD2DInputs[0], sampleUV);
	color *= gapAlpha; // premultiplied: zero out the gap (and antialias its edge) in gap mode
	fragColor = TDOutputSwizzle(color);
}
