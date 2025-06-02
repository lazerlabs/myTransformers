import matplotlib.pyplot as plt
import numpy as np
import os

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization - simple plotting like original iTransformer
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(name), exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(true, label='GroundTruth', linewidth=2, color='green')
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2, color='red', linestyle='--')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title('Stock Price Prediction')
    plt.savefig(name, bbox_inches='tight')
    plt.close()  # Close to prevent memory issues 
