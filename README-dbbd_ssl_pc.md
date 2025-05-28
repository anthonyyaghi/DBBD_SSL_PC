# DBBD_SSL_PC
## Data Preparation
### Download and unzip
You can either download the preprocessed data directly from [here](https://huggingface.co/datasets/Pointcept/scannet-compressed), go to `files and versions` tab, and then download `scannet.tar.gz`

Then, unzip the file in the `data` folder inside the root of the repository.

### Downsample the data
Next, we need to downsample the dataset to a specific number of points, while removing those that do not have enough points.
To do so, run the `processing_scripts/downsample.py` file. 
Command template: 
```sh
python processing_scripts/downsample.py --root_dir /path/to/dataset --output_dir /path/to/output --max_points 30000
 ```

Example:
```bash
python processing_scripts/downsample.py --root_dir ./data/scannet_original --output_dir ./data/scannet --max_points 30000
```


## Build the Solution

To build the solution, run the following command from the repository's root directory
```sh
docker build . -f ./Dockerfile -t pointcept-dbbd
```

## Run the Solution

To run the solution, run the following command from the repository's root directory
```sh
docker run -p 5678:5678 --gpus all --privileged -v /path_to_pointcept/Pointcept:/app/Pointcept -it pointcept-dbbd
```

## Running the training
After downsampling the dataset, you can now run the training.
you can now run the training command in the root of the repository:
```bash
sh scripts/train.sh -g 1 -d bmw -c inmind_spunet_point_level -n inmind_point_level
```
`-g` **1**: 1 gpu
 
`-d` **bmw**: Config folder
 
`-c` **inmind_spunet_point_level**: Name of .py config file, adjust file as needed and save it before training command.
 
`-n` **inmind_point_level**: You can name this whatever you want as it will be used to log the experiment results in `exp/Config folder` (in this example it is `exp/bmw`)` folder.

**Note**: 
- To use multi-gpu, always use `batch-size = 2*num_gpus`, this is to avoid any errors when code is run.
