# star_dataset
This repository contains codes for generating simulative light curve datasets which are used in the paper "NFD: Towards Real-Time Mining of Short Timescale Gravitational Microlensing Events from Variable Stars". Currently, three simulative datasets can be generated using this repository according to the experiment setting of the paper, namely full, gwac and threshold.   
## Example usage  
```
python gen_temp.py --dataset full # generating templates first  
python gen_data.py --dataset full # generating light curves then  
``` 
