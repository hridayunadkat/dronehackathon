# Project Ideas
- Multi drone coordination challenge - ROBIN
- AR pilot challenge _SAHA
- Obstacle avoidance with drones  -TIM
- UX challenge - RYAN
- Robotics localization - ANEESH
- Balloon challenge - ERIK


## Edge Systems
- Vision.. distribute video and track objects, because your enemy is time
  - Problem is camouflaged targets... tanks, soldiers, etc
  - Trawl OSINT data sources, quantization and use of the DSP for model inference, synthetic data for rare targets.. data collection from testing... jamil or danylo
- Video pipeline --> tracking system --> user / ground control --> detection --> visual tracker --> autonomus controls
  - On device detection ... identify target before you get jammed or it identifies you 
- On-device Machine Learning (ML)
- Radio Communications
  - Long range frequency hopping mesh communications to be unaffected by current Ukrainan, Russian, and American jammers...robust comms at ranges between 100 km to drone.
  - Custom network layer, custom ECC adaptive hoping, routing
  - Frequency hopping radio
  - Dual LoRa modem

## Sensory Intelligence
- Radio Detection Finding
- Object Tracking
- Object Detection
- Visual Positioning - aneesh, creates GPS-denied navigation to allow one pilot to send out multiple drones.
  - State estimation with visual-inertial odometry and satellite tile matching (comparing data against googole maps)... state estimation and then controller to fly through path

## Autonomy
- Guidance and Control - Tim talks about GNC for targeting moving targets, beehive management system, flying to multiple locations
- Vision-based Navigation

## Human Command and Control
- Ground Control Software - Saha... buildn the tech between users and comapny... automatic target recognition & behavior planning, but then also a heads up display (HUD) and target adjustment and controller interface
  Augmented reality interface challenge

## CONDOR Balloon
- Erik balloon team... evan and kevin are balloon programs... low cost, smart software, and effects and a lot of low cost balloons to deliver drones at long range distances (range of cruise missile)
  - 6 hours, 5-20 km altitude... fill system and altitude software.
  - Integrated software platform for mission planning... what days can we fly on... where do we need to move to hit targets of interest... target analysis... live/historical tracking... command and control
  - Launch plannning: launch sims and target analysis
  - Fleet management: live/historical tracking, command and control, steering, sim and targeting 


## Cloud Intelligence
- Beehive Intelligence System (Front End)
- Simulation of digital twins ofteh full system to implement training and evaluate behaviors
- Monitor a fleet of drones to:
  - Plan missions with the drones
  - Handle lower-level hardware tasks
  - Visualize a group of drones in a UI
  - Use the UI to plan missions
  - Abstract and identify potential target areas within the UI


SDK Guide: https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf
Usefel Repo with examples on how to use Tello API: https://github.com/damiafuentes/DJITelloPy/tree/master
