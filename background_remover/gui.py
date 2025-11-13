"""Tkinter based desktop application for background removal."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .remover import BackgroundRemover


class BackgroundRemovalApp:
    """Interactive GUI for removing image backgrounds."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("AI Background Remover")
        self.root.geometry("640x480")
        self.root.minsize(500, 400)

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._remover = BackgroundRemover()

        self._input_image: Optional[Image.Image] = None
        self._output_image: Optional[Image.Image] = None
        self._preview_photo: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        button_frame = ttk.Frame(container)
        button_frame.pack(fill=tk.X, pady=(0, 8))

        self.load_button = ttk.Button(
            button_frame, text="Open Image…", command=self._choose_image
        )
        self.load_button.pack(side=tk.LEFT, padx=(0, 8))

        self.process_button = ttk.Button(
            button_frame,
            text="Remove Background",
            command=self._process_image,
            state=tk.DISABLED,
        )
        self.process_button.pack(side=tk.LEFT, padx=(0, 8))

        self.save_button = ttk.Button(
            button_frame,
            text="Save Result…",
            command=self._save_image,
            state=tk.DISABLED,
        )
        self.save_button.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Select an image to begin.")
        status_label = ttk.Label(container, textvariable=self.status_var)
        status_label.pack(fill=tk.X, pady=(0, 8))

        preview_container = ttk.LabelFrame(container, text="Preview", padding=8)
        preview_container.pack(fill=tk.BOTH, expand=True)

        self.preview_label = ttk.Label(preview_container, anchor="center")
        self.preview_label.pack(fill=tk.BOTH, expand=True)

    # ----------------------------------------------------------------- events
    def _choose_image(self) -> None:
        filetypes = [
            ("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"),
            ("PNG", "*.png"),
            ("JPEG", "*.jpg *.jpeg"),
            ("Bitmap", "*.bmp"),
            ("All files", "*.*"),
        ]
        filename = filedialog.askopenfilename(
            title="Select image",
            filetypes=filetypes,
        )
        if not filename:
            return

        try:
            self._input_image = Image.open(filename)
            self._output_image = None
            self._update_preview(self._input_image)
            self.status_var.set("Image loaded. Click 'Remove Background'.")
            self.process_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.DISABLED)
        except Exception as exc:  # pragma: no cover - defensive UI handling
            messagebox.showerror("Error", f"Could not open image: {exc}")

    def _process_image(self) -> None:
        if self._input_image is None:
            return

        self.status_var.set("Processing…")
        self.process_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.load_button.config(state=tk.DISABLED)

        def task(image: Image.Image) -> Image.Image:
            return self._remover.remove_background(image)

        future = self._executor.submit(task, self._input_image.copy())
        threading.Thread(target=self._wait_for_future, args=(future,), daemon=True).start()

    def _wait_for_future(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:  # pragma: no cover - defensive UI handling
            self.root.after(0, self._handle_error, exc)
            return
        self.root.after(0, self._handle_success, result)

    def _handle_error(self, exc: Exception) -> None:
        self.status_var.set("Failed to process image.")
        self.load_button.config(state=tk.NORMAL)
        self.process_button.config(state=tk.NORMAL)
        messagebox.showerror("Background Removal", str(exc))

    def _handle_success(self, result: Image.Image) -> None:
        self._output_image = result
        self._update_preview(result)
        self.status_var.set("Background removed! You can save the result.")
        self.load_button.config(state=tk.NORMAL)
        self.process_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)

    def _save_image(self) -> None:
        if self._output_image is None:
            return

        default_name = "background_removed.png"
        filename = filedialog.asksaveasfilename(
            title="Save PNG",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG", "*.png"), ("All files", "*.*")],
        )
        if not filename:
            return

        try:
            self._output_image.save(Path(filename), format="PNG")
            self.status_var.set(f"Saved result to {filename}.")
        except Exception as exc:  # pragma: no cover - defensive UI handling
            messagebox.showerror("Save Error", f"Could not save file: {exc}")

    def _update_preview(self, image: Image.Image) -> None:
        max_width, max_height = 480, 320
        preview = image.copy()
        preview.thumbnail((max_width, max_height), Image.LANCZOS)

        if preview.mode != "RGBA":
            preview = preview.convert("RGBA")

        self._preview_photo = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self._preview_photo)

    def _on_close(self) -> None:
        self._executor.shutdown(wait=False)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    BackgroundRemovalApp().run()


if __name__ == "__main__":
    main()
