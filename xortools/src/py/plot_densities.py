import matplotlib
import matplotlib.pyplot as plt
import os
import tempfile
import numpy as np
from tqdm.auto import tqdm
import imageio.v2 as imageio
import textwrap


teal = np.array([54, 117, 136]) / 255


# Make an animated GIF file for the density evolution
def plot_densities(
    fname,
    out_fname,
    auto_stop=True,
    num_repeats_first_frame=10,
):
    with imageio.get_writer(out_fname, mode="I", fps=5, loop=0) as writer:
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Temporary directory created at:", temp_dir)
            prev_density = None
            num_imgs = 0
            with open(fname, "r") as infile:
                line = infile.readline()
                # First get through the comments portion
                comments = []
                while line.startswith("#"):
                    comments.append(line[1:].strip())
                    line = infile.readline()
                # Then extract the first vector of quantized llr values
                quantized_llrs = np.array(list(map(float, line.strip().split())))
                print(quantized_llrs.shape)
                for i, line in tqdm(list(enumerate(infile))):
                    if not line.strip():
                        break
                    density = np.array(list(map(float, line.strip().split())))
                    if prev_density is not None and (prev_density == density).all():
                        print("early auto stop")
                        break

                    prev_density = density
                    fig = plt.figure(figsize=(15, 15))
                    plt.grid(which="both")
                    plt.plot(
                        quantized_llrs,
                        density / (quantized_llrs[1] - quantized_llrs[0]),
                        color="gray",
                    )
                    plt.fill_between(
                        quantized_llrs,
                        density / (quantized_llrs[1] - quantized_llrs[0]),
                        color=teal,
                        alpha=0.8,
                    )
                    plt.xlabel("log-likelihood ratio")
                    plt.ylabel("probability density")
                    plt.title(
                        textwrap.fill("; ".join(comments), width=120) + "\n"
                        f"LLR density after {str(i).zfill(3)} sum-product iterations"
                    )
                    plt.ylim(1e-15, 1e2)
                    plt.yscale("log")
                    img_fname = os.path.join(temp_dir, f"plot{str(i).zfill(10)}.png")
                    plt.savefig(img_fname, dpi=128)
                    plt.clf()
                    plt.close(fig)
                    image = imageio.imread(img_fname)
                    writer.append_data(image)
                    num_imgs += 1
                    if num_imgs == 1:
                        for j in range(num_repeats_first_frame - 1):
                            writer.append_data(image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("plot some llr densities")
    parser.add_argument(
        "--fname",
        type=str,
        required=True,
        help="The density file. Each line should be a vector of doubles "
        "represented as space-delimited decimal values. "
        "The first line should be the quantized LLR values. "
        "Each line thereafter should be the density at 1 iteration "
        "of the density evolution.",
    )
    parser.add_argument(
        "--out-fname",
        type=str,
        required=True,
        help=".gif file where the plot should go",
    )
    args = parser.parse_args()
    plot_densities(args.fname, args.out_fname)
