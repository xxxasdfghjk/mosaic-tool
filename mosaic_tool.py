import os
import cv2
import numpy as np
import pyautogui
import sys
from tkinter import (
    Tk,
    Canvas,
    Button,
    filedialog,
    Frame,
    Scrollbar,
    Scale,
    Label,
    BOTTOM,
    StringVar,
)
from PIL import Image, ImageTk


class MosaicTool:
    def __init__(self, root, path=None):
        self.root = root
        self.root.title("Mosaic Tool")

        self.window_width = 1280
        self.window_height = 796
        self.root.geometry(f"{self.window_width}x{self.window_height}")

        # UI Components
        self.canvas_frame = Frame(root)
        self.canvas_frame.pack(side="left", fill="both", expand=True)

        self.canvas = Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill="both", expand=True)

        self.sidebar = Frame(root, width=200)
        self.sidebar.pack(side="right", fill="y")

        self.open_button = Button(
            self.sidebar, text="Open Folder", command=self.load_folder
        )
        self.open_button.pack(pady=10)

        self.save_button = Button(
            self.sidebar, text="Save & Next", command=self.save_and_next
        )
        self.save_button.pack(pady=10)

        self.apply_mosaic_button = Button(
            self.sidebar, text="Apply Mosaic", command=self.apply_mosaic
        )
        self.apply_mosaic_button.pack(pady=10)

        self.pixel_size_label = Label(self.sidebar, text="Mosaic Pixel Size")
        self.pixel_size_label.pack(pady=5)

        self.pixel_size_scale = Scale(self.sidebar, from_=2, to=50, orient="horizontal")
        self.pixel_size_scale.set(9)
        self.pixel_size_scale.pack(pady=5)

        self.brush_size_label = Label(self.sidebar, text="Brush Size")
        self.brush_size_label.pack(pady=5)

        self.brush_size_scale = Scale(
            self.sidebar, from_=1, to=100, orient="horizontal"
        )
        self.brush_size_scale.set(50)  # Default brush size
        self.brush_size_scale.pack(pady=5)

        self.undo_button = Button(self.sidebar, text="Undo", command=self.undo)
        self.undo_button.pack(pady=10)

        self.redo_button = Button(self.sidebar, text="Redo", command=self.redo)
        self.redo_button.pack(pady=10)

        self.page_label = Label(self.sidebar, text="Magnification")
        self.page_label.pack(pady=5)

        self.magnification_buttons = [
            Button(
                self.sidebar,
                text="33%",
                command=lambda: self.change_magnification(0.33),
            ),
            Button(
                self.sidebar, text="50%", command=lambda: self.change_magnification(0.5)
            ),
            Button(
                self.sidebar,
                text="100%",
                command=lambda: self.change_magnification(1.0),
            ),
        ]

        for btn in self.magnification_buttons:
            btn.pack(pady=5)

        self.image_list = []
        self.image_index = 0
        self.current_image = None
        self.original_image = None
        self.mask_layer = None
        self.draw_data = []
        self.redo_stack = []
        self.undo_stack = []  # To keep track of undo history
        self.scale_factor = 0.5
        # Initialize brush size from the scale
        self.brush_size = self.brush_size_scale.get()
        self.last_position = None

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.start_paint)
        self.canvas.bind("<ButtonRelease-1>", self.end_paint)
        self.canvas.bind("<Motion>", self.hover)  # Add hover event
        self.root.bind("<KeyPress>", self.key_event)
        self.root.bind("<KeyRelease>", self.on_key_release)

        self.is_dragging = False
        self.hover_mask_layer = np.zeros((1, 1), dtype=np.uint8)  # Temporary hover mask

        self.brush_size_scale.bind("<Motion>", self.update_brush_size)

        self.magnification_buttons = [
            Button(
                self.sidebar, text="50%", command=lambda: self.change_magnification(0.5)
            ),
            Button(
                self.sidebar,
                text="75%",
                command=lambda: self.change_magnification(0.75),
            ),
            Button(
                self.sidebar, text="100%", command=lambda: self.change_magnification(1)
            ),
            Button(
                self.sidebar,
                text="150%",
                command=lambda: self.change_magnification(1.5),
            ),
            Button(
                self.sidebar, text="200%", command=lambda: self.change_magnification(2)
            ),
        ]
        self.page_text = StringVar()
        self.page_text.set("0 / 0")
        self.page = Label(self.sidebar, textvariable=self.page_text).pack(side=BOTTOM)
        self.page_label = Label(self.sidebar, text="Page").pack(side=BOTTOM)

        self.image_id = None
        self.is_painting = False
        if not path is None:
            self.root.after(100, lambda: self.open_folder(path))

    def update_current_page(self, offset):
        if (
            self.image_index + offset < len(self.image_list)
            and self.image_index + offset >= 0
        ):
            self.image_index = self.image_index + offset
            self.page_text.set(
                f"{str(self.image_index + 1)} / {str(len(self.image_list))}"
            )

    def key_event(self, event):
        key = event.keysym
        if key == "Left":
            self.update_current_page(-1)
            self.load_image()
        if key == "Right":
            self.update_current_page(1)
            self.load_image()
        if key == "f":
            self.apply_mosaic()
        if key == "o":
            self.load_folder()
        if key == "z":
            self.undo()
        if key == "s":
            self.save_and_next()
        if key == "d":
            x, y = pyautogui.position()
            if not self.is_dragging:  # 初回押下時のみクリック
                pyautogui.mouseDown(x=x, y=y)
                self.is_dragging = True
        if key == "bracketleft":
            self.change_brush_size(5)
            self.display_image()

        if key == "bracketright":
            self.change_brush_size(-5)
            self.display_image()

    def change_brush_size(self, delta):
        # 現在のブラシサイズを変更
        new_brush_size = self.brush_size + delta
        # 範囲内に収める
        if 1 <= new_brush_size <= 100:
            self.brush_size = new_brush_size
            self.brush_size_scale.set(self.brush_size)

    def on_key_release(self, event):
        if self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False

    def update_brush_size(self, event):
        self.brush_size = (
            self.brush_size_scale.get()
        )  # Update brush size from the scale

    def open_folder(self, folder):
        if os.path.exists(folder) and os.path.isdir(folder):
            self.image_list = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(("png", "jpg", "jpeg"))
            ]
            self.image_index = 0
            self.update_current_page(0)
            self.load_image()

    def load_folder(self):
        folder = filedialog.askdirectory()
        self.open_folder(folder)

    def load_image(self):
        if self.image_index < 0 or self.image_index >= len(self.image_list):
            return

        file_path = self.image_list[self.image_index]
        image = cv2.imread(file_path)
        self.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_image = self.original_image.copy()
        self.mask_layer = np.zeros(
            (self.current_image.shape[0], self.current_image.shape[1]), dtype=np.uint8
        )
        self.hover_mask_layer = np.zeros(
            (self.current_image.shape[0], self.current_image.shape[1]), dtype=np.uint8
        )
        self.display_image()

    def display_image(self):
        if self.current_image is None:
            return

        scaled_image = cv2.resize(
            self.current_image,
            None,
            fx=self.scale_factor,
            fy=self.scale_factor,
            interpolation=cv2.INTER_NEAREST,
        )
        scaled_mask = cv2.resize(
            self.mask_layer,
            None,
            fx=self.scale_factor,
            fy=self.scale_factor,
            interpolation=cv2.INTER_NEAREST,
        )
        scaled_hover_mask = cv2.resize(
            self.hover_mask_layer,
            None,
            fx=self.scale_factor,
            fy=self.scale_factor,
            interpolation=cv2.INTER_NEAREST,
        )

        combined = scaled_image.copy()
        mask_overlay = (scaled_mask > 0) * 127
        combined[:, :, 0] = np.clip(combined[:, :, 0] + mask_overlay, 0, 255)
        combined[:, :, 1] = np.clip(combined[:, :, 1] - mask_overlay // 2, 0, 255)
        combined[:, :, 2] = np.clip(combined[:, :, 2] + mask_overlay, 0, 255)

        if self.hover_mask_layer is not None:
            hover_mask = (scaled_hover_mask > 0) * 127
            combined[:, :, 0] = np.clip(combined[:, :, 0] + hover_mask, 0, 255)
            combined[:, :, 1] = np.clip(combined[:, :, 1] + hover_mask // 2, 0, 255)
            combined[:, :, 2] = np.clip(combined[:, :, 2] + hover_mask, 0, 255)

        image = Image.fromarray(combined)
        self.tk_image = ImageTk.PhotoImage(image=image)
        # Canvasの幅と高さを取得
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # 中央揃えの座標を計算
        x = canvas_width // 2
        y = canvas_height // 2

        self.canvas.delete("all")
        self.image_id = self.canvas.create_image(
            x, y, anchor="center", image=self.tk_image
        )

        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def hover(self, event):
        if self.current_image is None:
            return

        x, y = event.x, event.y
        (lu, lb, ru, rb) = self.corner_points()

        if x < lu[0] or x > ru[0] or y < lu[1] or y > rb[1]:
            return
        x_rate = 1 - (ru[0] - x) / (ru[0] - lu[0])
        y_rate = 1 - (rb[1] - y) / (rb[1] - lu[1])
        (image_width, image_height) = self.image_size()
        circle_center_x = int(x_rate * image_width)
        circle_center_y = int(y_rate * image_height)

        self.hover_mask_layer = np.zeros(
            (self.current_image.shape[0], self.current_image.shape[1]), dtype=np.uint8
        )

        # Define the hover area to be the area around the cursor
        cv2.circle(
            self.hover_mask_layer,
            (circle_center_x, circle_center_y),
            self.brush_size,
            255,
            -1,
        )
        self.display_image()

    def corner_points(self):
        center_x, center_y = self.canvas.coords(self.image_id)
        image_width = self.tk_image.width()
        image_height = self.tk_image.height()

        return (
            (center_x - image_width / 2, center_y - image_height / 2),
            (center_x - image_width / 2, center_y + image_height / 2),
            (center_x + image_width / 2, center_y - image_height / 2),
            (center_x + image_width / 2, center_y + image_height / 2),
        )

    def image_size(self):
        height, width, channels = self.original_image.shape
        return (width, height)

    def paint(self, event):

        if self.current_image is None or not self.is_painting:
            return

        x = int(event.x)
        y = int(event.y)
        (lu, lb, ru, rb) = self.corner_points()

        if x < lu[0] or x > ru[0] or y < lu[1] or y > rb[1]:
            return

        x_rate = 1 - (ru[0] - x) / (ru[0] - lu[0])
        y_rate = 1 - (rb[1] - y) / (rb[1] - lu[1])
        (image_width, image_height) = self.image_size()
        circle_center_x = int(x_rate * image_width)
        circle_center_y = int(y_rate * image_height)

        # Update the hover mask to track mouse position during drag
        self.hover_mask_layer = np.zeros(
            (self.current_image.shape[0], self.current_image.shape[1]), dtype=np.uint8
        )
        cv2.circle(
            self.hover_mask_layer,
            (circle_center_x, circle_center_y),
            self.brush_size,
            255,
            -1,
        )

        if self.last_position:
            prev_x, prev_y = self.last_position
            cv2.line(
                self.mask_layer,
                (prev_x, prev_y),
                (circle_center_x, circle_center_y),
                255,
                self.brush_size * 2,
            )  # Adjust thickness
        # Update the mask layer (real drawing)
        cv2.circle(
            self.mask_layer,
            (circle_center_x, circle_center_y),
            self.brush_size_scale.get(),
            255,
            -1,
        )

        self.display_image()
        self.last_position = (circle_center_x, circle_center_y)

    def start_paint(self, event):
        if self.current_image is None:
            return

        x = int(event.x)
        y = int(event.y)
        (lu, lb, ru, rb) = self.corner_points()

        if x < lu[0] or x > ru[0] or y < lu[1] or y > rb[1]:
            return

        self.is_painting = True
        x_rate = 1 - (ru[0] - x) / (ru[0] - lu[0])
        y_rate = 1 - (rb[1] - y) / (rb[1] - lu[1])
        (image_width, image_height) = self.image_size()
        circle_center_x = int(x_rate * image_width)
        circle_center_y = int(y_rate * image_height)

        # Update the hover mask layer for click event
        self.hover_mask_layer = np.zeros(
            (self.current_image.shape[0], self.current_image.shape[1]), dtype=np.uint8
        )
        cv2.circle(
            self.hover_mask_layer,
            (circle_center_x, circle_center_y),
            self.brush_size,
            255,
            -1,
        )

        # Save the initial state for undo
        self.paint_start_mask = self.mask_layer.copy()
        # Also update the mask layer to apply the click
        cv2.circle(
            self.mask_layer,
            (circle_center_x, circle_center_y),
            self.brush_size,
            255,
            -1,
        )

        self.display_image()

        self.last_position = (circle_center_x, circle_center_y)

    def end_paint(self, event):
        if self.is_painting:
            # After finishing the paint, save the full mask change as one operation in undo stack
            self.undo_stack.append(
                ("mask", self.paint_start_mask, self.mask_layer.copy())
            )
            self.is_painting = False  # Stop painting
            self.paint_start_mask = None  # Clear the start mask
        self.last_position = None

    def apply_mosaic(self):
        if self.current_image is None or self.mask_layer is None:
            return

        self.undo_stack.append(
            ("mosaic", self.current_image.copy(), self.mask_layer.copy())
        )

        pixel_size = self.pixel_size_scale.get()

        # Create a copy to ensure only masked regions are processed
        mosaic_image = self.current_image.copy()

        # Get the bounding box of the mask
        mask_indices = np.where(self.mask_layer > 0)
        if len(mask_indices[0]) == 0:
            return

        y_min, y_max = np.min(mask_indices[0]), np.max(mask_indices[0])
        x_min, x_max = np.min(mask_indices[1]), np.max(mask_indices[1])

        for y in range(y_min, y_max, pixel_size):
            for x in range(x_min, x_max, pixel_size):
                y_end = min(y + pixel_size, y_max)
                x_end = min(x + pixel_size, x_max)

                # Check if the region is within the mask
                if np.any(self.mask_layer[y:y_end, x:x_end] > 0):
                    region = mosaic_image[y:y_end, x:x_end]
                    if region.size > 0:
                        avg_color = region.mean(axis=(0, 1)).astype(int)
                        mosaic_image[y:y_end, x:x_end] = avg_color

        # Apply the mosaic to the current image
        self.current_image = np.where(
            self.mask_layer[:, :, None] > 0, mosaic_image, self.current_image
        )
        self.mask_layer.fill(0)  # Clear the mask
        self.display_image()

    def save(self):
        if not self.image_list or self.current_image is None:
            return
        save_path = self.image_list[self.image_index]
        cv2.imwrite(save_path, cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))

    def save_and_next(self):
        if not self.image_list or self.current_image is None:
            return
        self.apply_mosaic()
        save_path = self.image_list[self.image_index]
        cv2.imwrite(save_path, cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
        self.update_current_page(1)
        self.load_image()

    def change_magnification(self, scale_factor):
        self.scale_factor = scale_factor
        self.display_image()

    def undo(self):
        if not self.undo_stack:
            return

        # Pop the last action from undo stack
        last_action = self.undo_stack.pop()

        if last_action[0] == "mosaic":
            # Undo mosaic: restore the image and mask to the previous state
            self.current_image = last_action[1]
            self.mask_layer = last_action[2]
            self.display_image()
        elif last_action[0] == "mask":
            # Undo mask: restore the previous mask state
            self.mask_layer = last_action[1]
            self.display_image()

    def redo(self):
        pass  # Implement redo logic


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    root = Tk()
    app = MosaicTool(root, arg)
    root.mainloop()
