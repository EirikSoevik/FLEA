# Fish Locomotion Extraction Analysis

FLEA is a tool allowing scientists to extract midlines from videos of swimming fish using deep learning in python. This is done by finding the fish outline, extracting the midlines and analysing the results. AI is used to automatically extract the outline.

# Designed for use with docker

Using deep learning requires many modules and packages which can be tedious and difficult to download in a matching set. Different hardware might require different versions, making installation hard. Therefore, several docker images will be made available for download so that one needs only identify the CUDA compatibility of one's own PC before downloading a correspoding image. Alternatively, a CPU version will also be made available. 

## Input:
- video of fish swimming. Must be filmed from above, and move along the x-axis. 

## Output:
- frequency spectrum using FFT, with dominating frequency
- travelling index
- midline movement for a predetermined set of 20 points, in time and space (size can be changed)

## Features:
- Deep learning using Detectron2, which is MetaAI's state of the art deep learning network.
- CUDA compatible - utilize your graphics card to greatly reduce computation time. In the authors own case, training of a model was 17 times faster with the GPU than with CPU
- CPU compatible - for those who can't or don't want to spend the time optimizing for GPU support
- 

## Planned improvements
- added mass identification, so that locomotion can be decomposed into two parts, one from the added mass and one without


## Use

1. Use inference.py to find segmentation masks for each frame 
2. Use midline_extraction.py to extract the midlines from each frame
3. Use midline_analysis.py to analyse the midlines
