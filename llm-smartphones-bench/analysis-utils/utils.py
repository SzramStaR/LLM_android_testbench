# Class for draggable text annotations
class DraggableText:
    def __init__(self, text_obj, x, y, threshold, ax, color, use_connector=True):
        self.text = text_obj
        self.figure = getattr(text_obj, "figure", None)
        self.canvas = (
            getattr(self.figure, "canvas", None) if self.figure is not None else None
        )
        self.x = x
        self.y = y
        self.threshold = threshold
        self.ax = ax
        self.color = color
        self.use_connector = use_connector
        self.press = None
        self.line = None

        if self.canvas is not None:
            self.cidpress = self.canvas.mpl_connect("button_press_event", self.on_press)
            self.cidrelease = self.canvas.mpl_connect(
                "button_release_event", self.on_release
            )
            self.cidmotion = self.canvas.mpl_connect(
                "motion_notify_event", self.on_motion
            )
        else:
            self.cidpress = self.cidrelease = self.cidmotion = None

    def update_line(self):
        """Update the line connecting text label to point"""
        if not self.use_connector:
            if (
                self.text.figure is not None
                and getattr(self.text.figure, "canvas", None) is not None
            ):
                self.text.figure.canvas.draw_idle()
            return

        anchor_x, anchor_y = self.text.get_position()
        try:
            renderer = self.text.figure.canvas.get_renderer()
            if renderer is None:
                self.text.figure.canvas.draw()
                renderer = self.text.figure.canvas.get_renderer()

            bbox = self.text.get_window_extent(renderer=renderer)
            px_disp, py_disp = self.ax.transData.transform((self.x, self.y))
            cx = min(max(px_disp, bbox.xmin), bbox.xmax)
            cy = min(max(py_disp, bbox.ymin), bbox.ymax)

            if abs(cx - bbox.xmin) < abs(cx - bbox.xmax) and (
                bbox.xmin <= cx <= bbox.xmin + 0.2
            ):
                cx -= 0.2
            elif abs(cx - bbox.xmax) <= abs(cx - bbox.xmin) and (
                bbox.xmax - 0.2 <= cx <= bbox.xmax
            ):
                cx += 0.2
            if abs(cy - bbox.ymin) < abs(cy - bbox.ymax) and (
                bbox.ymin <= cy <= bbox.ymin + 0.2
            ):
                cy -= 0.2
            elif abs(cy - bbox.ymax) <= abs(cy - bbox.ymin) and (
                bbox.ymax - 0.2 <= cy <= bbox.ymax
            ):
                cy += 0.2

            anchor_x, anchor_y = self.ax.transData.inverted().transform((cx, cy))
        except Exception:
            anchor_x, anchor_y = self.text.get_position()

        if self.line is None:
            self.line = self.ax.plot(
                [self.x, anchor_x],
                [self.y, anchor_y],
                color=self.color,
                alpha=0.5,
                linewidth=0.8,
                zorder=1,
            )[0]
        else:
            self.line.set_data([self.x, anchor_x], [self.y, anchor_y])
        self.text.figure.canvas.draw_idle()

    def on_press(self, event):
        """Handle clicks: left-click to start drag; right-click to delete annotation."""
        if event.inaxes != self.text.axes:
            return
        contains, _ = self.text.contains(event)
        if not contains:
            return
        if getattr(event, "button", None) == 3:
            try:
                if self.line is not None:
                    self.line.remove()
                    self.line = None
                self.text.remove()
            finally:
                canvas = self.canvas
                self.disconnect()
                if canvas is not None:
                    canvas.draw_idle()
            return
        if getattr(event, "button", None) == 1:
            self.press = (
                self.text.get_position()[0],
                self.text.get_position()[1],
                event.xdata,
                event.ydata,
            )

    def on_motion(self, event):
        """Move text as mouse is dragged"""
        if self.press is None or event.inaxes != self.text.axes:
            return

        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.text.set_position((x0 + dx, y0 + dy))
        self.update_line()

    def on_release(self, event):
        """Reset press when mouse is released"""
        self.press = None

    def disconnect(self):
        """Disconnect all callbacks"""
        if self.canvas is not None:
            if self.cidpress is not None:
                self.canvas.mpl_disconnect(self.cidpress)
            if self.cidrelease is not None:
                self.canvas.mpl_disconnect(self.cidrelease)
            if self.cidmotion is not None:
                self.canvas.mpl_disconnect(self.cidmotion)
        self.cidpress = self.cidrelease = self.cidmotion = None
