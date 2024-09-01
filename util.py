import numpy as np
from matplotlib import widgets
from matplotlib.transforms import TransformedPatchPath
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

__all__ = [
    'set_aspect_equal_3d',
    'make_simplex_triangles',
    'StateButton',
    'SliderStyle0',
]

def set_aspect_equal_3d(ax):
    """
    Sets Axes3D to use equal scale for all dimensions. Unfortunate that there's
    not an easy function already present in mplot3d.

    Parameters
    ----------
    ax : Axes3D
        The Axes3D to equalize.

    Returns: none
    """
    ax.set_box_aspect(aspect=(1,1,1))
    extent = np.array([
        ax.get_xlim(),
        ax.get_ylim(),
        ax.get_zlim(),
    ])
    wid = 0.5 * np.max(extent[:,1] - extent[:,0])
    ctr = 0.5 * (extent[:,1] + extent[:,0])
    ax.set_xlim(ctr[0]-wid, ctr[0]+wid)
    ax.set_ylim(ctr[1]-wid, ctr[1]+wid)
    ax.set_zlim(ctr[2]-wid, ctr[2]+wid)


def make_simplex_triangles(granularity):
    """
    Creates a mesh of the simplex in 3D

    granularity is an int, the larger it is the finer the mesh

    returns vectors, triangles
    """
    def make_triangulation(point_arr, triangle_arr=None, refine_level=1):
        def refine(point_arr, ind_arr, offset):
            new_point_arr = [
                0.5 * (point_arr[ind_arr[0]] + point_arr[ind_arr[1]]),
                0.5 * (point_arr[ind_arr[1]] + point_arr[ind_arr[2]]),
                0.5 * (point_arr[ind_arr[2]] + point_arr[ind_arr[0]]),
            ]
            new_triangle_arr = [
                (offset+2, ind_arr[0], offset+0),
                (offset+0, ind_arr[1], offset+1),
                (offset+1, ind_arr[2], offset+2),
                (offset+0, offset+1, offset+2),
            ]
            return new_point_arr, new_triangle_arr
        if triangle_arr is None:
            triangle_arr = [(0, 1, 2)]
        for _ in range(refine_level):
            old_triangle_arr = triangle_arr
            triangle_arr = []
            for triangle in old_triangle_arr:
                new_point_arr, new_triangle_arr = refine(
                    point_arr,
                    triangle,
                    len(point_arr)
                )
                point_arr += new_point_arr
                triangle_arr += new_triangle_arr
        return point_arr, triangle_arr
    point_arr = [
        np.array([1., 0, 0]),
        np.array([0., 1, 0]),
        np.array([0., 0, 1]),
    ]
    point_arr, triangle_arr = make_triangulation(point_arr, refine_level=granularity)
    vectors = np.stack(point_arr, axis=0)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    triangles = np.array(triangle_arr, dtype=np.int32)
    return vectors, triangles

class StateButton(widgets.Button):
    def __init__(self, ax, label_data_arr, *args, **kwargs):
        tmp = label_data_arr[0]
        super().__init__(ax, tmp if isinstance(tmp, str) else tmp['text'], *args, **kwargs)
        assert not hasattr(self, '_state')
        assert all(isinstance(_, str) for _ in label_data_arr) or all(isinstance(_, dict) for _ in label_data_arr)
        self.label_data_arr = label_data_arr
        self._state = 0
        self.on_clicked(self.increment_state)
    def get_state(self):
        return self._state
    def set_state(self, state):
        self._state = state
        label_data = self.label_data_arr[self._state]
        if isinstance(label_data, str):
            self.label.set_text(label_data)
        else:
            self.label.set_text(label_data['text'])
            self.label.set_color(label_data['color'])
        self.canvas.draw_idle()
        return self
    def increment_state(self, event=None):
        self.set_state((self._state + 1) % len(self.label_data_arr))

class SliderStyle0(widgets.Slider):
    def __init__(self, fontsize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert 'horizontal' == self.orientation
        self.track.set(x=0., y=0., width=1., height=0.5)
        kwargs_parent = {k: v for k, v in kwargs.items() if k not in {
            'ax', 'label', 'valmin', 'valmax', 'valinit', 'valfmt',
            'closedmin', 'closedmax', 'slidermin', 'slidermax', 'dragging',
            'valstep', 'orientation', 'initcolor', 'track_color',
            'handle_style',
        }}
        self.poly.remove()
        self.poly = self.ax.axvspan(self.valmin, self.valinit, 0., 0.5, **kwargs_parent)
        initcolor = 'r'
        if 'initcolor' in kwargs:
            initcolor = kwargs['initcolor']
        self.vline.remove()
        self.vline = self.ax.axvline(self.valinit, 0, 1, color=initcolor, lw=1,
                                    clip_path=TransformedPatchPath(self.track))
        # NOTE: Fixing the ball with self._handle is too finnicky, so I don't
        # worry about it
        self._handle.set_zorder(3)
        self.label.set_fontsize(fontsize)
        self.label.set_verticalalignment('top')
        self.label.set_horizontalalignment('left')
        self.label.set_position((0, 1))
        self.valtext.set_fontsize(fontsize)
        self.valtext.set_verticalalignment('top')
        self.valtext.set_horizontalalignment('right')
        self.valtext.set_position((1, 1))
    def set_val(self, val):
        super().set_val(val)
        xy = self.poly.xy
        xy[2] = val, 0.5
        xy[3] = val, 0.
        self._handle.set_xdata([val])
        self.poly.xy = xy

