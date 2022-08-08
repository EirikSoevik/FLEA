# Fish Locomotion Extraction Analysis

FLEA is a tool allowing scientists to extract midlines from videos of swimming fish using deep learning in python. This is done by finding the fish outline, extracting the midlines and analysing the results. AI is used to automatically extract the outline.

# Download dockerimages for easy installation and use

Using deep learning requires many modules and packages which can be tedious and difficult to download in a matching set. Different hardware might require different versions, making installation hard. Therefore, several docker images will be made available for download so that one needs only identify the CUDA compatibility of one's own PC before downloading a correspoding image. Alternatively, a CPU version will also be made available. 

## Input:
- video of fish swimming. Must be filmed from above, and move along the x-axis. 

## Output:
- discretized midline, in 2D space and time
- analysis of midline motion, including frequency spectrum using FFT and travelling index

## Features:
- Deep learning using Detectron2, which is MetaAI's state of the art deep learning network.
- CUDA compatible - utilize your graphics card to greatly reduce computation time. In the authors own case, training of a model was 17 times faster with the GPU than with CPU
- CPU compatible - for those who can't or don't want to spend the time optimizing for GPU support


# See the wiki for more in-depth guides on installation and use
https://github.com/EirikSoevik/FLEA/wiki

