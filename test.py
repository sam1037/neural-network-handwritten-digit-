import tkinter as tk

def on_mouse_down(event):
    canvas.delete("all")  # Clear the canvas
    canvas.create_line(event.x, event.y, event.x, event.y, width=5, fill='black')
    canvas.tag_bind("current_line", "<B1-Motion>", on_mouse_move)

def on_mouse_move(event):
    canvas.create_line(event.x, event.y, event.x, event.y, width=5, fill='black', tags="current_line")

def on_mouse_up(event):
    canvas.tag_unbind("current_line", "<B1-Motion>")
    # Process the user's drawing here
    # You can access the coordinates of the drawing using:
    start_x = event.x
    start_y = event.y
    end_x = canvas.canvasx(event.x)
    end_y = canvas.canvasy(event.y)

root = tk.Tk()
canvas = tk.Canvas(root, width=400, height=400)
canvas.pack()

canvas.bind("<Button-1>", on_mouse_down)
canvas.bind("<ButtonRelease-1>", on_mouse_up)

root.mainloop()