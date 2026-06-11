import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from PIL import Image, ImageTk
import numpy as np
import os

class EuroSATGuiApp:
    def __init__(self, window, checkpoint_path='./checkpoints/best_model.pth'):
        self.window = window
        self.window.title("EuroSAT Satellite Image Classifier")
        self.window.geometry("500x650")
        self.window.configure(bg="#f0f0f0")
        
        # Handle cleanup elegantly when the user clicks the X close button
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 1. Initialize computing device and Model weights
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(checkpoint_path):
            messagebox.showerror("Error", f"Could not find model checkpoint at:\n{checkpoint_path}\nPlease run train.py first!")
            self.window.destroy()
            return
            
        print("Loading prediction engine model weights...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.class_names = checkpoint['class_names']
        
        # Import the model architecture dynamically
        from src.model import EuroSATCNN
        self.model = EuroSATCNN(num_classes=checkpoint['config']['model']['num_classes']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 2. Build GUI Layout Components
        self.create_widgets()

    def create_widgets(self):
        # Top Title Title Frame
        title_label = tk.Label(self.window, text="Satellite Land Cover Lookup", font=("Helvetica", 16, "bold"), bg="#f0f0f0", fg="#333333")
        title_label.pack(pady=15)
        
        # Main Interactive Selection Button
        self.select_btn = tk.Button(self.window, text="Choose Satellite Picture", command=self.load_and_predict_image, font=("Helvetica", 11, "bold"), fg="white", bg="#4A90E2", padx=10, pady=5, relief="flat")
        self.select_btn.pack(pady=10)
        
        # Picture Display Window Canvas Box
        self.image_label = tk.Label(self.window, text="No Image Selected", font=("Helvetica", 10), bg="#e0e0e0", relief="sunken")
        self.image_label.pack(pady=15)
        
        # Bottom Results Section Display Block
        self.result_frame = tk.LabelFrame(self.window, text=" Classification Results ", font=("Helvetica", 11, "bold"), bg="#f0f0f0", padx=15, pady=15)
        self.result_frame.pack(fill="x", padx=30, pady=15)
        
        self.class_label = tk.Label(self.result_frame, text="Predicted Class: ---", font=("Helvetica", 12), bg="#f0f0f0", anchor="w")
        self.class_label.pack(fill="x", pady=2)
        
        self.conf_label = tk.Label(self.result_frame, text="Confidence Level: ---", font=("Helvetica", 12), bg="#f0f0f0", anchor="w")
        self.conf_label.pack(fill="x", pady=2)

    def load_and_predict_image(self):
        # Trigger native operating system file explorer lookup UI dialog
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
        )
        
        if not file_path:
            return # User backed out or cancelled selection
            
        try:
            # --- Rendering logic for GUI Window Canvas ---
            raw_image = Image.open(file_path).convert('RGB')
            # Scale image up for display purposes so it's readable in the window
            display_image = raw_image.resize((256, 256), Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(display_image)
            
            self.image_label.configure(image=tk_image, text="")
            self.image_label.image = tk_image # Retain garbage collector reference layout
            
            # --- Machine Learning Engine Processing logic ---
            # Process image to match training parameters down to 64x64
            ml_input_img = raw_image.resize((64, 64))
            img_array = np.array(ml_input_img).astype(np.float32) / 255.0
            img_array = np.transpose(img_array, (2, 0, 1))
            
            img_tensor = torch.tensor(img_array).unsqueeze(0) # Add mandatory batch dimension shape [1, 3, 64, 64]
            
            # Standard ImageNet normalization matching validation pipeline
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            img_tensor = img_tensor.to(self.device)
            
            # Evaluate Image Tensor Array values
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence, pred_idx = torch.max(probabilities, 0)
                
            predicted_class = self.class_names[pred_idx.item()]
            confidence_percentage = confidence.item() * 100
            
            # Update Dashboard GUI string values interactively
            self.class_label.configure(text=f"Predicted Class:  {predicted_class}", fg="#2C3E50")
            self.conf_label.configure(text=f"Confidence Level: {confidence_percentage:.2f}%", fg="#27AE60" if confidence_percentage > 75 else "#E67E22")
            
        except Exception as e:
            messagebox.showerror("Processing Failure", f"An error occurred loading image:\n{str(e)}")

    def on_closing(self):
        # Stops loop processes and exits when window framework closes
        print("Safely terminating application interface.")
        self.window.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = EuroSATGuiApp(root, checkpoint_path='./checkpoints/best_model.pth')
    root.mainloop()