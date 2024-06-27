import pyift.pyift as ift
import matplotlib.pyplot as plt

import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil

import cv2
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from torchvision import transforms

###############################################3
##################################################3    
##############################################
# Importing the datasets
#############################################
##############################################
################################################


def normalize_filename(filename):
    """
    Normalize the filename to ensure it has a consistent number format.
    
    Args:
    filename (str): The filename to normalize.
    
    Returns:
    str: The normalized filename.
    """
    base_name = filename.split('.')[0]
    normalized_base_name = f"{int(base_name):06}"  # 6 digits padding - Necessary because the names of the ground truth images are different to the ones of te dataset
    return normalized_base_name + '.png'

def rename_files_in_directory(directory_path):
    """
    Rename all .png files in the specified directory to have a six-digit padded filename.
    
    Args:
    directory_path (str): The path to the directory containing the files to be renamed.
    """
    for filename in os.listdir(directory_path):
        if filename.endswith('.png'):
            normalized_filename = normalize_filename(filename)
            original_file_path = os.path.join(directory_path, filename)
            new_file_path = os.path.join(directory_path, normalized_filename)
            if original_file_path != new_file_path:
                os.rename(original_file_path, new_file_path)
                print(f"Renamed '{filename}' to '{normalized_filename}'")


def setup_image_paths(folder_path, ground_truth_path):
    """
    Prepares and validates the paths to image files and their corresponding ground truth masks.
    Returns tuple containing lists of image paths and ground truth paths.
    Raises ValueError if the lists of images and ground truth files do not match.
    """
    # Ensure filenames are consistent across directories
    rename_files_in_directory(folder_path)
    rename_files_in_directory(ground_truth_path)

    # Get sorted lists of file names
    image_files = sorted(os.listdir(folder_path))
    ground_truth_files = sorted(os.listdir(ground_truth_path))

    if image_files != ground_truth_files:
        raise ValueError("Image files do not match ground truth files.")

    # Build full paths
    image_paths = [os.path.join(folder_path, fname) for fname in image_files]
    ground_truth_paths = [os.path.join(ground_truth_path, fname) for fname in ground_truth_files]

    return image_paths, ground_truth_paths, image_files, ground_truth_files

  
def plot_images_in_grid(image_paths, image_titles, rows=2, cols=3, figsize=(15, 10)):
    """
    Plots images in a grid layout.

    Args:
    image_paths (list of str): List of paths to the images.
    image_titles (list of str): List of titles for the images.
    rows (int): Number of rows in the image grid. Default is 2.
    cols (int): Number of columns in the image grid. Default is 3.
    figsize (tuple): Figure dimension (width, height) in inches. Default is (15, 10).

    Returns:
    None
    """
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = np.array(axs)  
    axs = axs.reshape((rows, cols))  
    
    count = 0
    num_images = len(image_paths)
    
    # Ensure we do not exceed the list length
    max_images = rows * cols
    
    # Plotting the images from the list
    for i in range(rows):
        for j in range(cols):
            if count < min(num_images, max_images):
                img = Image.open(image_paths[count])
                axs[i, j].imshow(img, cmap='gray')  
                axs[i, j].set_title(image_titles[count])
                count += 1
            else:
                axs[i, j].axis('off') 
    
    plt.tight_layout()
    plt.show()
    

def load_images_as_tensors(image_paths, is_grayscale=False):
    """
    Load images from the given list of image paths and transform them into tensors.
    
    Args:
    image_paths (list of str): List of paths to the image files.
    is_grayscale (bool): Whether the images are grayscale. Defaults to False.
    
    Returns:
    list of torch.Tensor: List of loaded images as tensors.
    """
    if is_grayscale:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    images = [transform(Image.open(image_path)) for image_path in image_paths]
    return images


def plot_image_tensors(train_input_images, train_ground_truth_images, image_names, index=0):
    """
    Plot the training data and corresponding ground truth images.

    Args:
    train_input_images (list of torch.Tensor): List of input image tensors for training.
    train_ground_truth_images (list of torch.Tensor): List of ground truth image tensors for training.
    image_names (list of str): List of image file names.
    index (int): The index of the image to plot. Defaults to 0.
    """
    input_image = train_input_images[index].permute(1, 2, 0).numpy()
    ground_truth_image = train_ground_truth_images[index].squeeze().numpy()
    image_name = image_names[index]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(input_image,cmap='gray')
    axes[0].set_title(f'Input Image: {image_name}')
    axes[0].axis('off')
    
    axes[1].imshow(ground_truth_image, cmap='gray')
    axes[1].set_title(f'Ground Truth Image: {image_name}')
    axes[1].axis('off')
    
    plt.show()

def copy_model_files(src_dir, dest_exp_dir, delete_existing=False):
    """
    Copy all files from the source directory to the specified experiment folder.
    Optionally delete the contents of the destination directory before copying.

    Args:
    src_dir (str): Path to the source directory containing the files to copy.
    dest_exp_dir (str): Path to the destination experiment directory where files will be copied.
    delete_existing (bool): Whether to delete existing contents of the destination directory before copying. Defaults to False.
    """
    if delete_existing:
        if os.path.exists(dest_exp_dir):
            for filename in os.listdir(dest_exp_dir):
                file_path = os.path.join(dest_exp_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        else:
            os.makedirs(dest_exp_dir)
    
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_exp_dir, item)
        
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dest_path)
    
    print(f"All files from {src_dir} have been copied to {dest_exp_dir}")

def load_and_plot_images(image_folder, save_folder):
    """
    Load all images from the specified folder, plot them with titles, and save the plot.

    Args:
    image_folder (str): Path to the folder containing the images to load and plot.
    save_folder (str): Path to the folder where the plot will be saved.
    """
    # Create the save folder if it does not exist
    os.makedirs(save_folder, exist_ok=True)

    # Get list of all image files in the directory
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    
    # Determine the number of rows and columns for the plot grid
    num_images = len(image_files)
    num_cols = 4  # Number of columns in the plot grid
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate rows needed
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    
    for idx, image_file in enumerate(image_files):
        img_path = os.path.join(image_folder, image_file)
        img = Image.open(img_path)
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(image_file, fontsize=18)
        axes[idx].axis('off')
    
    # Hide any remaining empty subplots
    for i in range(idx + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout(pad=0.1)

    # Save the plot
    save_path = os.path.join(save_folder, 'Flim_Decoder_TrainingData_Output.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()



###############################################3
##################################################3    
##############################################
# Thresholding the pupil
#############################################
##############################################
################################################

def perform_image_operations(image_path, threshold=10, A1=15.0, A2=30.0):
    """
    Perform a series of morphological operations on an image.

    Args:
    image_path (str): Path to the image file.
    threshold (int): Threshold value for the binarization process.
    A1 (float): Radius for the circular structuring element used in the first dilation and erosion.
    A2 (float): Radius for the circular structuring element used in the second erosion and dilation.

    Returns:
    dict: Dictionary containing all intermediate images.
    """
    results = {}
    
    # Load the original image
    img1 = ift.ReadImageByExt(image_path)
    results['original'] = img1

    # Apply threshold
    bin = ift.Threshold(img1, 0, threshold, 255)
    results['threshold'] = bin

    # Apply dilation
    A = ift.Circular(A1)
    dil = ift.Dilate(bin, A, None)
    results['dilate1'] = dil

    # Apply erosion (dilation + erosion = closing operation)
    erode = ift.Erode(dil, A, None)
    results['erode1'] = erode

    # Apply erosion 
    A = ift.Circular(A2)
    erode2 = ift.Erode(erode, A, None)
    results['erode2'] = erode2

    # Final dilation (erosion + dilation = opening operation)
    dil2 = ift.Dilate(erode2, A, None)
    results['dilate2'] = dil2

    return results

def plot_image_operations(results, image_file, num_rows=3, num_cols=2, figsize=(12,12)):
    """
    Plot the results of image processing operations, converting image objects to plot format.

    Args:
    results (dict): Dictionary containing image objects from various processing steps.
    image_file (str): Image file name for title annotations.
    num_rows (int): Number of rows in the subplot grid.
    num_cols (int): Number of columns in the subplot grid.
    figsize (tuple): Figure size.

    Returns:
    None
    """
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.ravel()
    
    titles = ['Original', 'Threshold', 'Dilate1', 'Erode1', 'Erode2', 'Dilate2']
    for i, key in enumerate(titles):
        image_to_plot = results[key.lower()].ToPlot() if hasattr(results[key.lower()], 'ToPlot') else results[key.lower()]
        axs[i].imshow(image_to_plot, cmap='gray')
        axs[i].set_title(f"{image_file} - {key}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()
    
    
def create_red_mask(mask_matrix):
    """Create a red transparency mask from a binary mask matrix."""
    red_mask = np.zeros((mask_matrix.shape[0], mask_matrix.shape[1], 4), dtype=np.uint8)
    red_indices = mask_matrix > 0  # Where the mask is not zero
    red_mask[red_indices] = [255, 0, 0, 100]  # Red with transparency
    return Image.fromarray(red_mask)

def overlay_mask_on_image(image, mask_image):
    """Overlay a red transparency mask on an image."""
    combined = Image.alpha_composite(image.convert('RGBA'), mask_image)
    return combined

def adjust_ground_truth(image_path):
    """Adjust the ground truth image to have 255 intensity for visible areas since the ground truth only has 0s and 1s."""
    ground_truth = Image.open(image_path)
    ground_truth_array = np.array(ground_truth) * 255
    return Image.fromarray(ground_truth_array, mode='L')

def display_images_with_masks(original_image_paths, ground_truth_image_paths, num, dil2):
    """Plot the original image and the ground truth image with overlays."""
    
    original_image_path = original_image_paths[num] 
    ground_truth_image_path = ground_truth_image_paths[num]
    mask = dil2.AsNumPy().reshape((480, -1))/255
    
    original_image = Image.open(original_image_path).convert('RGBA')
    adjusted_ground_truth_image = adjust_ground_truth(ground_truth_image_path).convert('RGBA')

    red_mask_image = create_red_mask(mask)
    
    combined_image = overlay_mask_on_image(original_image, red_mask_image)
    combined_ground_truth = overlay_mask_on_image(adjusted_ground_truth_image, red_mask_image)

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    axs[0].imshow(combined_image)
    axs[0].set_title('Mask over Original Image')
    axs[1].imshow(combined_ground_truth)
    axs[1].set_title('Mask over Ground Truth')

    plt.tight_layout()
    plt.show()
    
    
def plot_all_images(image_paths, num_rows=20, num_cols=5, threshold=10, A1=15.0, A2=30.0):
    """Plot all images in a grid layout with overlays from operations."""
    
    # Define the directory for saving 'dil2' results, which is the mask
    dil2_dir = 'datasets/pupil_mask'
    os.makedirs(dil2_dir, exist_ok=True)  # Ensure the directory exist
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 80))  
    axs = axs.ravel()  # Flatten the axis array

    hyperparam_config = {
        'config1': (6, 15.0, 20.0), # Threshold, Closing operation circular adjacency value, Opening operation circular adjacency value
        'config2': (6, 17.0, 31.0),
        'config3': (10, 15.0, 30.0), # Default 
        'config4': (15, 22.0, 25.0),
        'config5': (18, 25.0, 29.0)
    }


    image_with_hyperparam_config1 = {7, 8, 9, 55, 69, 70, 84, 85, 86, 87, 88, 89, 90, 91, 97}
    image_with_hyperparam_config2 = {10,13,57,98}
    image_with_hyperparam_config4 = {72, 73, 74, 75,78,79,92,93,100}
    image_with_hyperparam_config5 = {71}
    
    
    for idx, image_path in tqdm(enumerate(image_paths),total=len(image_paths)):
        config = hyperparam_config['config3'] # Default configuration
         
        if idx in image_with_hyperparam_config1:
            config = hyperparam_config['config1']
        elif idx in image_with_hyperparam_config2:
            config = hyperparam_config['config2']
        elif idx in image_with_hyperparam_config4:
            config = hyperparam_config['config4']
        elif idx in image_with_hyperparam_config5:
            config = hyperparam_config['config5']
        
        threshold, A1, A2 = config
        
        # Perform operations on each image
        results = perform_image_operations(image_path=image_path, threshold=threshold, A1=A1, A2=A2)
        dil2 = results['dilate2']
        
        dil2_normalized = (dil2.AsNumPy())
        dil2_image = Image.fromarray(dil2_normalized.astype(np.uint8))  # Convert to image, ensure dtype is uint8
        filename = os.path.basename(image_path)  # Extract filename from the path
        save_path = os.path.join(dil2_dir, filename)  # Create the full save path
        dil2_image.save(save_path)  # Save the image
        
        mask = dil2.AsNumPy().reshape((480, -1))
        red_mask_image = create_red_mask(mask)

        # Composite the red mask with the original image
        original_image = Image.open(image_path).convert('RGBA')
        combined_image = overlay_mask_on_image(original_image, red_mask_image)

        # Plotting the combined image
        axs[idx].imshow(combined_image)
        image_chosen = os.path.basename(image_path)
        axs[idx].set_title('Mask over Image ' + image_chosen)
        axs[idx].axis('off')

    plt.savefig('Figures/100_images_pupil_masks.png')
    
    plt.tight_layout()
    plt.show()
    
###############################################3
##################################################3    
##############################################
# Creating the seeds from the masked pupil
#############################################
##############################################
################################################

def process_image_to_extract_ring(mask_path, dilation_size1, dilation_size2,ring_with_pupil=True):
    """
    Process an image to extract a ring-shaped region by performing binary dilations and subtractions.

    Parameters:
        mask_path (str): Path to the binary mask image.
        dilation_size1 (float): The radius for the first dilation.
        dilation_size2 (float): The radius for the second dilation, should be larger than the first.
        ring_with_pupil (True or False): Used to return the ring mask with the pupil mask together.

    Returns:
        Image: The resulting ring-shaped image after processing.
    """
    # Read the mask image
    mask_im = ift.ReadImageByExt(mask_path)
    
    # Perform two dilations with different sizes
    mask_di1 = ift.DilateBin(mask_im, None, dilation_size1)
    mask_di2 = ift.DilateBin(mask_im, None, dilation_size2)
    
    # Subtract the original mask from the dilated masks to form two new masks
    m_sub1 = ift.Sub(mask_di1[0], mask_im)
    m_sub2 = ift.Sub(mask_di2[0], mask_im)
    
    # Subtract the smaller dilated mask from the larger dilated mask to get the ring
    mask_ring = ift.Sub(m_sub2, m_sub1)
    
    # Putting the pupil back in the mask
    if ring_with_pupil:
        mask_ring_with_pupil_mask = ift.Add(mask_ring,mask_im)
        return mask_ring_with_pupil_mask
    
    return mask_ring

def process_and_save_masks(mask_paths, pupil_masks, output_dir, dilation_size1=19.0, dilation_size2=26.0):
    """
    Processes a list of mask paths to extract certain features and save them to a specified directory.

    Args:
    mask_paths (list of str): List of paths to the mask files.
    pupil_masks (list of str): List of output filenames for the processed masks.
    output_dir (str): Directory where the processed masks will be saved.
    process_image_to_extract_ring (function): Function to process masks, configured with specific dilation sizes and options.
    dilation_size1 (float): First dilation size parameter for the processing function.
    dilation_size2 (float): Second dilation size parameter for the processing function.

    Returns:
    None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all mask paths
    for num, mask_path in tqdm(enumerate(mask_paths), total=len(mask_paths)):
        # Process each mask with specified dilation sizes and ring settings
        mask_ring_with_pupil_mask = process_image_to_extract_ring(
            mask_path,
            dilation_size1=dilation_size1,
            dilation_size2=dilation_size2,
            ring_with_pupil=True
        )
        
        # Create the full save path
        save_path = os.path.join(output_dir, pupil_masks[num])
        
        # Convert processed mask to a NumPy array and reshape if necessary
        mask_array = mask_ring_with_pupil_mask.AsNumPy()
        seed = mask_array.reshape(480, -1)  # Example reshape, adjust dimensions as necessary
        
        # Save the processed mask to the file
        seed_to_save = Image.fromarray(seed.astype(np.uint8))
        seed_to_save.save(save_path)



def plot_images_with_masks(image_paths, mask_paths):
    """
    Plots images in a grid with their corresponding masks overlaid.

    Args:
    image_paths (list of str): List of paths to the original images.
    mask_paths (list of str): List of paths to the mask images.
    process_image_to_extract_ring (function): Function to process masks to extract specific features.
    create_red_mask (function): Function to create a red transparency mask from a binary mask matrix.
    overlay_mask_on_image (function): Function to overlay a transparency mask on an image.

    Returns:
    None
    """

    # Setup the figure with 20 rows and 5 columns
    fig, axs = plt.subplots(20, 5, figsize=(20, 80))
    axs = axs.ravel()  # Flatten the axis array

    # Loop through each image and mask path, process, and plot them
    for num, mask_path in tqdm(enumerate(mask_paths), total=len(mask_paths)):
        original_image = Image.open(image_paths[num]).convert('RGBA')
        mask_ring_with_pupil_mask = process_image_to_extract_ring(mask_path,
                                                                  dilation_size1=19.0,
                                                                  dilation_size2=26.0,
                                                                  ring_with_pupil=True)

        red_seed_image = create_red_mask(mask_ring_with_pupil_mask.AsNumPy())
        combined_image = overlay_mask_on_image(original_image, red_seed_image)

        # Plotting the combined image
        axs[num].imshow(combined_image)
        image_chosen = os.path.basename(image_paths[num])
        axs[num].set_title('Seeds over Image ' + image_chosen)
        axs[num].axis('off')

    # Save the figure to a file and display it
    plt.savefig('Figures/100_images_seed_masks')
    plt.tight_layout()
    plt.show()

###############################################3
##################################################3    
##############################################
# Image Foresting Transform
#############################################
##############################################
################################################







###############################################3
##################################################3    
##############################################
# Results
#############################################
##############################################
################################################

def read_image(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) 
    if image is None:
        raise FileNotFoundError(f"Image at path {filepath} could not be found.")
    return image

def dice_coefficient(im1, im2):
    im1 = im1.AsNumPy().astype(np.bool_)
    im2 = im2.AsNumPy().astype(np.bool_)
    intersection = np.logical_and(im1, im2)
    return 2.0 * intersection.sum() / (im1.sum() + im2.sum())

def get_classification_metrics(img1, img2):

    cm = confusion_matrix(img1.flatten(), img2.flatten())
    report = classification_report(img1.flatten(), img2.flatten())

    return cm, report

def image_error(estimated_img, reference_img, filename,save=False):
    # Calculate absolute difference
    difference = np.abs(estimated_img.AsNumPy()/ 255 - reference_img.AsNumPy())
    
    # Calculate DICE coefficient
    dice_score = dice_coefficient(estimated_img, reference_img)
    
    # Setup the figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = [
        f'Estimated Image - {filename}', 
        f'Reference Image - {filename}', 
        f'Absolute Difference\nDICE Score: {dice_score:.4f}'
    ]
    images = [estimated_img.AsNumPy(), reference_img.AsNumPy(), difference]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()
    
    if save:
        plt.savefig('Figures/difference_between_image_'+ filename)


def compare_all_images_DICE(folder1, folder2,verbose=False):
    dice_scores = []
    image_names = []
    filenames = [f'{i:06d}.png' for i in np.arange(1,101,1)]  

    for filename in filenames:
        path1 = f'{folder1}/{filename}'
        path2 = f'{folder2}/{filename}'
        try:
            img1 = ift.ReadImageByExt(path1)
            img2 = ift.ReadImageByExt(path2)
            score = dice_coefficient(img1, img2)
            dice_scores.append(score)
            image_names.append(filename)
        except FileNotFoundError as e:
            print(e)
            continue  

    dice_mean = np.mean(dice_scores)
    dice_std = np.std(dice_scores)

    # Sorting scores and filenames together
    sorted_indices = np.argsort(dice_scores)
    sorted_filenames = np.array(image_names)[sorted_indices]
    sorted_scores = np.array(dice_scores)[sorted_indices]
    
    # Top 5 best and worst scores
    top_5_best_dice = list(zip(sorted_filenames[-5:][::-1], sorted_scores[-5:][::-1]))
    top_5_worst_dice = list(zip(sorted_filenames[:5], sorted_scores[:5]))

    if verbose:
        print("Top 5 Best DICE Scores and their corresponding image names:")
        for name, score in top_5_best_dice:
            print(f"{name}: {score:.4f}")

        print("\nTop 5 Worst DICE Scores and their corresponding image names:")
        for name, score in top_5_worst_dice:
            print(f"{name}: {score:.4f}")
    
    return dice_scores, dice_mean, dice_std,top_5_best_dice,top_5_worst_dice


def aggregate_confusion_matrix(folder1, folder2):
    """Aggregate confusion matrix from all images comparisons."""
    total_cm = np.zeros((2, 2), dtype=int)

    for i in range(1, 101):  
        img1 = read_image(f'{folder1}/{i:06d}.png')/255
        img2 = read_image(f'{folder2}/{i:06d}.png')

        cm = confusion_matrix(img1.flatten(), img2.flatten())
        total_cm += cm

    return total_cm