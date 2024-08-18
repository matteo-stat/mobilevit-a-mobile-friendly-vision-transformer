from matplotlib import pyplot as plt
from typing import Tuple, Dict, Optional
import datetime as dt

def plot_training_history(history: Dict, figsize: Tuple[int, int], show: bool = True, output_path: Optional[str] = None) -> None:
    """
    save a plot showing training history loss

    Args:
        history (Dict): history dictionary created after running keras method model.fit
        output_path (str): 
        figsize (Tuple[int, int]): plot size
        show (Optional[bool], optional): optionally show or not the plot. Defaults to True.
        output_path (Optional[str], optional): optional path where the plot image will be saved. Defaults to None.        
    """
    plt.figure(figsize=figsize)
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
    else:
        plt.title('Training Loss')
    plt.xlabel('Epoch')  
    plt.ylabel('Loss')
    plt.legend() 
    plt.grid(True)
    
    # save plot if required
    if output_path is not None:
        timestamp = dt.datetime.now().strftime('%Y%m%d-%H%M')
        plt.savefig(f'{output_path}/{timestamp}-training-history.png')

    # show plot if required
    if show:
        plt.show()
