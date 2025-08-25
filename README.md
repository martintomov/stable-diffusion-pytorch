# stable-diffusion-pytorch
![PyTorch Badge](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=fff&style=flat)
![Lightning Badge](https://img.shields.io/badge/Lightning-792EE5?logo=lightning&logoColor=fff&style=flat)

Personal implementation of Stable Diffusion in PyTorch from scratch. Educational and research purposes only.

<table align="center">
  <tr>
    <td style="text-align:center;">prompt = "A cat under the snow with blue eyes, covered by snow, highly detailed, realistic, ultra sharp, cinematic, 100mm lens, 8k resolution."</td>
    <td style="text-align:center;"><img src="img/results/notebook/demo_output.png" alt="demo_output" width="800"/></td>
  </tr>
</table>

## Text to Image
<table align="center">
  <tr>
    <td style="text-align:center;"><img src="img/results/demo/text-to-image-o1.png" alt="demo_output 1" width="800"/></td>
    <td style="text-align:center;"><img src="img/results/demo/text-to-image-o2.png" alt="demo_output 2" width="800"/></td>
    <td style="text-align:center;"><img src="img/results/demo/text-to-image-o3.png" alt="demo_output 3" width="800"/></td>
    <td style="text-align:center;"><img src="img/results/demo/text-to-image-o4.png" alt="demo_output 4" width="800"/></td>
  </tr>
</table>

## Image to Image
<table align="center">
  <tr>
    <td style="text-align:center;"><img src="img/results/noise/noise_level_0.png" alt="demo_output 1" width="800"/></td>
    <td style="text-align:center;"><img src="img/results/noise/noise_level_250.png" alt="demo_output 2" width="800"/></td>
    <td style="text-align:center;"><img src="img/results/noise/noise_level_750.png" alt="demo_output 3" width="800"/></td>
    <td style="text-align:center;"><img src="img/results/demo/image-to-image-o5.png" alt="demo_output 4" width="800"/></td>
  </tr>
</table>

## Getting started

1. **Create a Conda environment:**

    ```sh
    conda create -n sd-env python=3.11
    conda activate sd-env
    ```

2. **Clone the repository:**

    ```sh
    git clone https://github.com/martintomov/stable-diffusion-pytorch.git
    cd stable-diffusion-pytorch
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Download weights:**

    Download `v1-5-pruned-emaonly.ckpt` (4.27 GB) from [RunwayML on Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) and save it in the `/data` folder. Feel free to download any other weights you like.

5. **Run the demo:**

    You can run the demo via `demo.py` or `demo.ipynb` notebook by your preference. Both demo scripts are configured to use a MPS GPU device, which can be easily adjusted at the start of the code file.

    ```sh
    # To run the demo script
    python demo.py

    # Or open and run the Jupyter notebook
    jupyter notebook demo.ipynb
    ```

## Dependencies

- PyTorch
- Numpy
- tqdm
- Pillow
- transformers
- lightning
- ipykernel

## Happy experimenting! 🚀
