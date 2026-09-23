## DTD DEL interactive

DTD DEL is an interactive camera-powered installation in downtown Denver. 

Remote login w/DWService
or Rustdesk: 421 801 510 / D3ld35kt0p

## TODOs

#### Install:

- [ ] Get TD license and load on machine

#### General app:

- [ ] Add python extension to CameraFX tox
- [ ] Save individual viz as external saved tox files
- [x] Cycle through visuals on a timer - every 15 mins
  - [ ] Try to shut down inactive viz (via extension?)
- [ ] Fix perform window UI code - why isn't the AppStore event picking up?

#### Add ip camera stream

> Create a Video Stream In TOP operator in TouchDesigner.
> Enter the stream URL in the URL parameter using the format rtsp://username:password@cameraip:554/stream (adjust user, password, and port to match your device).
- https://www.hikvision.com/us-en/products/network-products/ip-ptz-cameras/value-series/ds-2de2a404iw-de3/
- https://assets.hikvision.com/prd/public/all/doc/m000045324/UD22754B-B_Network-Speed-Dome_User-Manual_E7%2C-5.6.10-11%2C-5.7.0_PDF1-TEST_en-US_20221221.PDF?token=cc3c8e6ff9d7c76c0d00397953877b8b&t=1789183637

Camera: 
Hikvision - DS-2DE2A404IW-DE3
https://www.hikvision.com/us-en/products/network-products/ip-ptz-cameras/value-series/ds-2de2a404iw-de3/

IP address:
- 192.168.0.20
Web admin: 
- http://192.168.0.20/

Login:
- admin / PCcam1984!

RTSP stream (for VLC and TouchDesigner):
- rtsp://admin:PCcam1984!@192.168.0.20:554/stream

Apps:
- HiToolsDelivery - this lets you find the camera on the network and do extra config & control
- AdvancedIPScanner - also lets you find devices on the network but doesn't tell you that it's the Hikvision camera


#### Fix in Yolo

- [ ] Improve bytetracker - why do boxes keep going fo so long after loss while we also see flickering boxes. These both shouldn't be true
- [ ] z-index of debug boxes in any viz - attach z to y coord, w/low muiltiplier
- [ ] Warmup still fails sometimes? How do we ensure that ONNX model initializes

#### Ambient interactive concepts:

- [ ] POPs plexus - https://www.youtube.com/watch?v=Dm6rU_1EVKI
- [ ] Raycast texture w/ reaction diffusion?
- [ ] move captured rects next to each other
- [ ] time particles w/past texture of people
- [ ] particles launching off people - stars, sparkles, happy things
- [x] Motion particles
- [x] segmentation of just people, but bounding boxes would be fine
- [x] labels that say nice things about people

#### Other, later

- Update kittredge project with latest haxlib techniques
