## Development Log

### Going into FDP and Report - November 20, 2024

#### Relevant FDP and Report Expectations

* Show how design meets initial criteria
    * Important initial criteria:
        * Identify deviations with 90% accuracy
        * Provide feedback to user
        * Minimize user input
        * Instill confidence that I will be able to implement the required functionality
* Show engineering tests to confirm the above
    * Need to develop these tests before FDP
    * Need a dataset
    * Need performance metrics
* Engineering judgement
    * Above
    * Already have done some research into a host of methods for object detection
        * Convey this research and these decisions more clearly
* Schedule moving forward
    * Need to have an idea of where I am vs. where I need to be
    * Need specific, attainable features that still need to be implemented

#### Software Aspects Blocking FDP and Report

* See above

#### The Plan

The first step will be to develop a more complete and usable dataset.

This dataset should allow for testing of how well the design meets the criteria mentioned above. The most important performance characteristics to show are:

* Variety of objects identified
* "Accuracy" with which the system can identify out-of-place objects (and reject in-place objects), subject to:
    * Varying lighting conditions
    * Presence of people
    * Other potential sources of error:
        * Slight variation in camera position -> No setup for this, push to after FDP and report
        * Noise

Images should be captured over the course of a full day of natural (and artificial) lighting, from an angle similar to that expected during use. Currently, sunrise is at 7:18 AM and sunset is at 4:52 PM.

There are two aspects to the data we expect to have access to:

* The baseline image(s)
* The image(s) for comparison

It is important that the dataset gives a good idea of what we can expect for both of these.

For the baseline, photos should be taken throughout the day, perhaps in short bursts. They should be taken in both artificial and natural lighting conditions. The baseline photos can be expected to have little to no variation between them in terms of object motion.

For the comparison image, it is a bit harder to say what is useful. A strong algorithm may rely on short bursts of video or camera motion. Camera motion is not achievable at this time. The photos should again be taken throughout the day. The RPi camera module 3 offers many opportunities for tweaking how the photos are taken and processed.

One valuable feature of the RPCM3 is phase detection autofocus (PDAF). This is a strength because it reduces user setup and facilitates strong performance in varying room geometries and camera positions. This feature does, however, have an associated risk. The autofocus might be unpredictable and might result in varying volumes of focus. Objects that fall outside of this volume of focus will appear more blurry. "Dumb" change detection approaches may be vulnerable to these changes. For now, the dynamic autofocus will be used to generate a test set, but if strange artifacts are observed across the dataset, this should be considered as one potential cause.

##### Baseline Dataset
The dataset needs to be made over the next couple of days. Tomorrow (21st), baseline photos will be taken for my room throughout the day. Photos will be captured from 7 am until 5 or 6 pm. The photos will be taken at maximum resolution (with potential for later downscaling). The photos will be taken once every 30 minutes, in ~1 second bursts of 30 frames.

TODO: In the future, datasets should be generated for multiple spaces. For now though, the dataset should be sufficient to illustrate the plan for testing and improvement.

##### Object Dataset
This dataset will need to be generated on a day where I am able to be home all day. Again, photos will be taken from sunrise to sunset. Every hour, a "photoshoot" will occur The following photos will be taken:
* 2 photos with person in frame (varying places) and various out-of-place objects.
* 1 photo with several out of place objects, no person and door closed. Additionally, some objects will have moved slightly from the "baseline" position. (bed, clothes hamper, etc.)
    * the slightly moved objects can be changed throughout the day, but do not need to be.

Action Items:
* Helper software for annotating dataset
* Software for evaluating performance

Out of place objects will be kept the same for the photos to avoid significant setup each hour.

##### Annotating the dataset
To assess the performance of the dataset, the dataset will need to be annotated. For the out of place objects, since they are in the same place for each photo, the same annotation can be used for all photos. Annotations will be generated manually by drawing boxes around all of the out of place objects. Detections will be assessed automatically by seeing how closely they match the drawn boxes*
\*this may change depending on how well this seems to assess the performance.

##### Evaluating performance
software performance will be evaluated based on the metrics discussed in my intelligent vehicle systems class. False positives, false negatives, true positives, and true negatives will be assessed.

### FDP Prep Continued - November 21, 2024
Baseline photos are being captured. On Saturday Nov 23 the dataset of comparison photos will be made. 

# Remaining blockers for FDP:
* Instill confidence that I will be able to meet the required functionality
    * Potentially demo current functionality
    * Identification accuracy
        * Current issues:
        | Issue | Description | Potential Remedies |
        |-------|-------------|--------------------|
        | Split object contours | Single objects are sometimes producing multiple detections | Dilation, IOU?, SAM? |
        | Unwanted Detections |  |  |
        | **Shadows and hilights** | Due to changes in lighting | SAM? Edge detection? Aspect Ratios? baseline throughout the day?|
        | Shifting of image | Due to positioning system | Adaptive positioning? Image transformation? Contour filtering? |
        | People and Animals | People and Animals show up as detections | Person/cat/dog detection CV and operating modes. |
        | **Deformable Objects** | e.g. couches, blinds, chairs | Colour? Texture Mapping? SAM? User Feedback? |
        | **Object Motion** | Objects that may be moved but are part of the baseline, e.g. chairs, blinds/curtains, decorations | Depending on the degree of motion the problem changes. We want to omit small motion for sure. | Texture segmentation or edges to get some sort of object boundaries and then comparison of contour properties (size, aspect ratio, location) SAM could also be used|
        | **Absence of objects** | Need to discuss with aaron whether this should be considered a problem. Is it useful to have it point out object absences? | Looking at edges, gradients, or textures | 
    * Updatable baseline
        * The device should be able to update the baseline to include new objects if indicated by the user. The process for this is only known at a high level.
        * Involves cropping out segments of the comparison photo and adding them to the baseline.
        * Updating the baseline will be important
        * There might be some way of learning to filter out detections in regions where false detections are common, for example due to moving scenery outside a window, deforming bed, etc.
        * Nice if baseline includes expected environmental lighting variations 
    * Provide _"closed loop feedback"_ to user by positioning
        * Need a solid plan/demonstration of how bounding boxes in the image will be transmitted to projector/positioning commands.
    * Operating mode with person detection