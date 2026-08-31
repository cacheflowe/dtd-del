## DTD DEL interactive

DTD DEL is an interactive camera-powered installation in downtown Denver. 

## TODOs

#### Install:

- [ ] Get Rustdesk installed on machine
- [ ] Get TD license and load on machine

#### General app:

- [ ] Add python extension to CameraFX tox
- [ ] Save individual viz as external saved tox files
- [x] Cycle through visuals on a timer - every 15 mins
  - [ ] Try to shut down inactive viz (via extension?)
- [ ] Fix perform window UI code - why isn't the AppStore event picking up?

#### Fix in Yolo

- [ ] Improve bytetracker - why do boxes keep going fo so long after loss while we also see flickering boxes. These both shouldn't be true
- [ ] z-index of debug boxes in any viz - attach z to y coord, w/low muiltiplier
- [ ] Warmup still fails sometimes? How do we ensure that ONNX model initializes

#### Ambient interactive concepts:

- [ ] POPs plexus - https://www.youtube.com/watch?v=Dm6rU_1EVKI
- [ ] move captured rects next to each other
- [ ] time particles w/past texture of people
- [ ] particles launching off people - stars, sparkles, happy things
- [x] Motion particles
- [x] segmentation of just people, but bounding boxes would be fine
- [x] labels that say nice things about people

#### Other, later

- Update kittredge project with latest haxlib techniques
