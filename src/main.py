import os
import numpy as np
import pandas as pd
import customtkinter as ctk
from PIL import Image, ImageTk
import tensorflow as tf
from tkinter import filedialog, messagebox

# Set the appearance mode and color theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class VegetableClassifierApp:
    def __init__(self, model_path='models/multi_class_classifier.keras'):
        """
        Initialize the Vegetable Classifier Application
        
        :param model_path: Path to the pre-trained TensorFlow model
        """
        # Main window setup
        self.root = ctk.CTk()
        self.root.title("🥦 Fruit & Vegetable Classifier")
        self.root.geometry("1000x800")
        
        # Load the model
        self.model = tf.keras.models.load_model(model_path)
        self.class_labels = ["apple", "cabbage", "carrot", "cucumber", "eggplant", "pear"]
        
        # Tracking results
        self.all_results = []
        self.upload_counter = 0
        
        # App UI setup
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface components"""
        # Main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Top section with buttons
        top_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        top_frame.pack(fill="x", pady=10)
        
        # Title
        title_label = ctk.CTkLabel(
            top_frame, 
            text="Fruit & Vegetable Classifier", 
            font=("Arial", 24, "bold")
        )
        title_label.pack(side="left", padx=20)
        
        # Upload Button
        self.upload_button = ctk.CTkButton(
            top_frame, 
            text="📤 Upload Images", 
            command=self.upload_images,
            fg_color="green",
            hover_color="darkgreen",
            width=150
        )
        self.upload_button.pack(side="right", padx=10)
        
        # Save Results Button (initially hidden)
        self.save_button = ctk.CTkButton(
            top_frame, 
            text="💾 Save All Results", 
            command=self.save_all_results,
            fg_color="blue",
            hover_color="darkblue",
            width=150
        )
        self.save_button.pack(side="right", padx=10)
        
        # Scrollable frame for all results
        self.results_scrollable_frame = ctk.CTkScrollableFrame(
            self.main_frame, 
            width=960, 
            height=600
        )
        self.results_scrollable_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
    def predict_image(self, image_path):
        """
        Predict the class of a single image
        
        :param image_path: Path to the image file
        :return: Predicted class and probability
        """
        try:
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = self.model.predict(img_array)
            predicted_class = self.class_labels[np.argmax(predictions)]
            probability = np.max(predictions)
            return predicted_class, probability
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            return None, None
        
    def upload_images(self):
        """Handle image upload and display"""
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Image files", "*.jpg;*.png;*.jpeg")]
        )
        
        if not file_paths:
            return
        
        # Increment upload counter
        self.upload_counter += 1
        
        # Create a batch frame for this upload
        batch_frame = ctk.CTkFrame(self.results_scrollable_frame)
        batch_frame.pack(pady=10, padx=5, fill="x")
        
        # Batch header
        batch_label = ctk.CTkLabel(
            batch_frame, 
            text=f"Upload Batch #{self.upload_counter} - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", 
            font=("Arial", 14, "bold")
        )
        batch_label.pack(side="left", padx=5)
        
        # Process and display each image
        for path in file_paths:
            # Predict the class and probability
            predicted_class, probability = self.predict_image(path)
            
            if predicted_class is None:
                continue
            
            # Prepare result data
            result = {
                "Upload Batch": self.upload_counter,
                "Timestamp": pd.Timestamp.now(),
                "Image": os.path.basename(path), 
                "Predicted Class": predicted_class, 
                "Probability": f"{probability:.2%}"
            }
            
            # Add to overall results
            self.all_results.append(result)
            
            # Create a frame for each image result
            result_frame = ctk.CTkFrame(batch_frame)
            result_frame.pack(pady=5, padx=5, fill="x")
            
            # Display image
            img = Image.open(path)
            img = img.resize((150, 150))
            img_tk = ImageTk.PhotoImage(img)
            
            image_label = ctk.CTkLabel(result_frame, image=img_tk)
            image_label.image = img_tk
            image_label.pack(side="left", padx=5)
            
            # Display prediction details
            details_frame = ctk.CTkFrame(result_frame, fg_color="transparent")
            details_frame.pack(side="left", expand=True, fill="both")
            
            filename_label = ctk.CTkLabel(
                details_frame, 
                text=f"File: {os.path.basename(path)}", 
                font=("Arial", 12)
            )
            filename_label.pack(anchor="w")
            
            class_label = ctk.CTkLabel(
                details_frame, 
                text=f"Predicted Class: {predicted_class}", 
                font=("Arial", 12, "bold"),
                text_color="green"
            )
            class_label.pack(anchor="w")
            
            prob_label = ctk.CTkLabel(
                details_frame, 
                text=f"Confidence: {probability:.2%}", 
                font=("Arial", 12)
            )
            prob_label.pack(anchor="w")
        
        # Show save results button if there are results
        if self.all_results:
            self.save_button.pack(side="right", padx=10)
        
    def save_all_results(self):
        """
        Save all prediction results to a single CSV file
        """
        if not self.all_results:
            messagebox.showwarning("No Results", "No images have been classified yet.")
            return
        
        # Default save location with timestamp
        default_filename = f"classifier_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv")]
        )
        
        if save_path:
            try:
                # Convert results to DataFrame for saving
                results_df = pd.DataFrame(self.all_results)
                
                # Save results to CSV
                results_df.to_csv(save_path, index=False)
                
                # Show success message
                messagebox.showinfo(
                    "Success", 
                    f"Results saved successfully!\n\nCSV: {save_path}"
                )
                
            except Exception as e:
                messagebox.showerror("Save Error", str(e))
    
    def run(self):
        """Run the application main loop"""
        self.root.mainloop()

# Create and run the application
if __name__ == "__main__":
    app = VegetableClassifierApp()
    app.run()
