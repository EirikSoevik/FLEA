#Fish Locomotion Extraction Analysis

AI is used to automatically extract the outline of fish from input videos, after which a midline is extracted and subsequently analysed. 

##Input:
- video of fish swimming. Must be filmed from above, and move along the x-axis. 

##Output:
- frequency spectrum using FFT, with dominating frequency
- travelling index
- midline movement for a _ set of 20 points, in time and space

##Features:
- Deep learning using Detectron2, which is MetaAI's state of the art deep learning network.
- CUDA compatible - utilize your graphics card to greatly reduce computation time. In the authors own case, training of a model was 17 times faster with the GPU than with CPU
- CPU compatible - for those who can't or don't want to spend the time optimizing for GPU support
- 
