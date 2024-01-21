
import textwrap
import functools
from dataclasses import dataclass, field

import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.raise_window'] = False
import matplotlib.pyplot as plt
from matplotlib import widgets
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.transforms import TransformedPatchPath

@dataclass
class RealSymmetricOperator3:
    eigenvalues: np.ndarray = field(default_factory=lambda: np.ones(3))
    eigenvectors: np.ndarray = field(default_factory=lambda: np.eye(3))
    def verify(self):
        if not (3,) == self.eigenvalues.shape:
            return False
        if not np.isrealobj(self.eigenvalues):
            return False
        if not (3, 3) == self.eigenvectors.shape:
            return False
        if not np.isrealobj(self.eigenvectors):
            return False
        if 1e-12 < np.max(np.abs(np.eye(3) - (self.eigenvectors @ self.eigenvectors.T))):
            return False
        if 0 > np.linalg.det(self.eigenvectors):
            return False
        return True
    def get_rayleigh(self, x):
        assert 3 == x.shape[-1]
        assert np.isrealobj(x)
        VTx = np.squeeze(self.eigenvectors.T @ x[..., None], -1)
        return np.sum(self.eigenvalues * (VTx ** 2), axis=-1)
    def get_basisless_rayleigh(self, x):
        assert 3 == x.shape[-1]
        return np.sum(self.eigenvalues * (x ** 2), axis=-1)

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
    triangles = np.array(triangle_arr, dtype=np.int32)
    return vectors, triangles

def axis_angle_to_matrix(v0, v1, v2, theta):
    assert isinstance(v0, float)
    assert isinstance(v1, float)
    assert isinstance(v2, float)
    assert isinstance(theta, float)
    v = np.array([v0, v1, v2])
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    theta *= np.pi
    return (
        np.outer(v, v) +
        (np.cos(theta) * (np.eye(3) - np.outer(v, v))) +
        (np.sin(theta) * np.array([[0, -v2, v1], [v2, 0, -v0], [-v1, v0, 0]]))
    )

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

class StateButton(widgets.Button):
    def __init__(self, ax, label_arr, *args, **kwargs):
        super().__init__(ax, label_arr[0], *args, **kwargs)
        assert not hasattr(self, '_state')
        self.label_arr = label_arr
        self._state = 0
        self.on_clicked(self.increment_state)
    def get_state(self):
        return self._state
    def increment_state(self, event=None):
        self._state += 1
        self._state %= len(self.label_arr)
        self.label.set_text(self.label_arr[self._state])
        self.canvas.draw_idle()

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
    #@override
    def set_val(self, val):
        super().set_val(val)
        xy = self.poly.xy
        xy[2] = val, 0.5
        xy[3] = val, 0.
        self._handle.set_xdata([val])
        self.poly.xy = xy

class App:

    def __init__(self):
        self.ops_colorwheel = (
            (1., 0., 0.),
            (0., 1., 0.),
            (0., 0., 1.),
        )
        self.num_ops = len(self.ops_colorwheel)
        self.translation_tables = {
            'mode-button': (
                'lemniscate',
                'pos_shell',
                'neg_shell',
            ),
            'reflections-button': (
                False,
                True,
            ),
        }
        self._refs_keep_alive = []
        self.registry = {
            'lamb-slider': [[] for _ in range(self.num_ops)],
            'axang-slider': [[] for _ in range(self.num_ops)],
            'mode-button': [None for _ in range(self.num_ops)],
            'reflections-button': None,
            'reflection-alpha': None,
        }
        fig = plt.figure()
        self.fig = fig
        subfigs0 = fig.subfigures(
            1, 3,
            wspace=0,
            hspace=0,
            width_ratios=[10, 1, 4],
        )
        self.subfigs0 = subfigs0
        subfigs1 = subfigs0[2].subfigures(
            2, 1,
            wspace=0,
            hspace=0,
            height_ratios=[10, 4],
        )
        self.subfigs1 = subfigs1
        mosaic = [
            ['swap-views', 'toggle-reflections', 'reflection-alpha', 'reflection-alpha'],
        ] + sum(([
            [f'op{op_id}lamb0', f'op{op_id}lamb0', f'op{op_id}axang0', f'op{op_id}axang0'],
            [f'op{op_id}lamb1', f'op{op_id}lamb1', f'op{op_id}axang1', f'op{op_id}axang1'],
            [f'op{op_id}lamb2', f'op{op_id}lamb2', f'op{op_id}axang2', f'op{op_id}axang2'],
            [f'op{op_id}mode',  f'op{op_id}mode',  f'op{op_id}axang3', f'op{op_id}axang3'],
        ] for op_id in range(self.num_ops)), start=[])
        axd = subfigs1[0].subplot_mosaic(mosaic)
        #axd = subfigs1[0].subplot_mosaic(
        #    [
        #        ['swap-views', 'toggle-reflections', 'add-operator', 'remove-operator'],
        #        ['op0lamb0', 'op0lamb0', 'op0axang0', 'op0axang0'],
        #        ['op0lamb1', 'op0lamb1', 'op0axang1', 'op0axang1'],
        #        ['op0lamb2', 'op0lamb2', 'op0axang2', 'op0axang2'],
        #        ['op1lamb0', 'op1lamb0', 'op1axang0', 'op1axang0'],
        #        ['op1lamb1', 'op1lamb1', 'op1axang1', 'op1axang1'],
        #        ['op1lamb2', 'op1lamb2', 'op1axang2', 'op1axang2'],
        #        ['op2lamb0', 'op2lamb0', 'op2axang0', 'op2axang0'],
        #        ['op2lamb1', 'op2lamb1', 'op2axang1', 'op2axang1'],
        #        ['op2lamb2', 'op2lamb2', 'op2axang2', 'op2axang2'],
        #    ],
        #)
        margin = 0.01
        self.rect_main = (margin, margin, 1-(2*margin), 1-(2*margin))
        axd['main'] = subfigs0[0].add_axes(self.rect_main, projection='3d')
        self.rect_alt = (margin, margin, 1-(2*margin), 1-(2*margin))
        axd['alt'] = subfigs1[1].add_axes(self.rect_alt)
        self.axd = axd
        #fig, axd = plt.subplot_mosaic(
        #    [
        #        ['main', 'swap-views', 'toggle-reflections', 'add-operator', 'remove-operator'],
        #        ['main', 'op0lamb0', 'op0lamb0', 'op0axang0', 'op0axang0'],
        #        ['main', 'op0lamb1', 'op0lamb1', 'op0axang1', 'op0axang1'],
        #        ['main', 'op0lamb2', 'op0lamb2', 'op0axang2', 'op0axang2'],
        #        ['main', 'op1lamb0', 'op1lamb0', 'op1axang0', 'op1axang0'],
        #        ['main', 'op1lamb1', 'op1lamb1', 'op1axang1', 'op1axang1'],
        #        ['main', 'op1lamb2', 'op1lamb2', 'op1axang2', 'op1axang2'],
        #        ['main', 'op2lamb0', 'op2lamb0', 'op2axang0', 'op2axang0'],
        #        ['main', 'op2lamb1', 'op2lamb1', 'op2axang1', 'op2axang1'],
        #        ['main', 'op2lamb2', 'op2lamb2', 'op2axang2', 'op2axang2'],
        #        ['main', 'alt', 'alt', 'alt', 'alt'],
        #    ],
        #    gridspec_kw={
        #        'width_ratios': [10, 1, 1, 1, 1],
        #        'height_ratios': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5],
        #    },
        #    per_subplot_kw={
        #        'main': {'projection': '3d'},
        #        'alt': {'yscale': 'log'}
        #    },
        #)
        #self.fig, self.axd = fig, axd

        # General
        fontsize = 7
        tmp = self._keep_alive(widgets.Button(axd['swap-views'], 'Swap Views'))
        tmp.label.set_fontsize(fontsize)
        tmp.on_clicked(self.swap_views)
        tmp = self._keep_alive(StateButton(axd['toggle-reflections'], (
            'Show Refls',
            'Hide Refls',
        )))
        self.registry['reflections-button'] = tmp
        tmp.label.set_fontsize(fontsize)
        tmp.on_clicked(self.toggle_reflections)
        tmp = self._keep_alive(SliderStyle0(
            fontsize, axd['reflection-alpha'], r'$\alpha$',
            0., 1., valinit=0.5,
        ))
        self.registry['reflection-alpha'] = tmp
        tmp.on_changed(self.on_change_reflection_alpha)

        # Per-Op
        fontsize = 7
        for op_id in range(self.num_ops):
            for dim in range(3):
                tmp = self._keep_alive(SliderStyle0(
                    fontsize, axd[f'op{op_id}lamb{dim}'], rf'$\lambda_{dim}(A_{op_id})$',
                    -10., 10., valinit=1.,
                ))
                self.registry['lamb-slider'][op_id].append(tmp)
                tmp.on_changed(functools.partial(self.update_lamb, op_id, dim))
            for dim in range(3):
                tmp = self._keep_alive(SliderStyle0(
                    fontsize, axd[f'op{op_id}axang{dim}'], rf'Axis$_{dim}(A_{op_id})$',
                    -1., 1., valinit=(1. if 0==dim else 0.),
                ))
                self.registry['axang-slider'][op_id].append(tmp)
                tmp.on_changed(functools.partial(self.update_axang, op_id, dim))
            tmp = self._keep_alive(SliderStyle0(
                fontsize, axd[f'op{op_id}axang3'], rf'$\theta(A_{op_id})/\pi$',
                -1., 1., valinit=0.,
            ))
            self.registry['axang-slider'][op_id].append(tmp)
            tmp.on_changed(functools.partial(self.update_axang, op_id, dim))
            tmp = self._keep_alive(StateButton(axd[f'op{op_id}mode'], (
                f'Showing $A_{op_id}$ as\nLemniscate',
                f'Showing $A_{op_id}$ as\n+Shell',
                f'Showing $A_{op_id}$ as\n-Shell',
            )))
            self.registry['mode-button'][op_id] = tmp
            tmp.label.set_fontsize(fontsize)
            tmp.on_clicked(functools.partial(self.toggle_op_mode, op_id))


        # Initial Cofiguration
        self.ax3D = axd['main']
        self.ax2D = axd['alt']

        self.init_data()
        self.on_data_changed(*range(self.num_ops))

        # Fix layout issues
        plt.pause(0.1)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def _keep_alive(self, x):
        self._refs_keep_alive.append(x)
        return x

    def swap_views(self, event):
        def get_info(ax):
            info = {}
            if isinstance(ax, Axes3D):
                info['class'] = Axes3D
                info['ax.elev'] = ax.elev
                info['ax.azim'] = ax.azim
                info['ax.roll'] = ax.roll
                # method taken from matplotlib source code for
                # matplotlib/lib/mpl_toolkits/mplot3d/axes3d.py:shareview
                info['ax._vertical_axis'] = {0: "x", 1: "y", 2: "z"}[ax._vertical_axis]
                info['ax._focal_length'] = ax._focal_length
                # It's easier to just remember how to plot data then deal with
                # figuring out how to recover it directly from the Axes3D, so
                # we do that with self.render_data
            else:
                info['class'] = Axes
            return info
        def consume_info(info, subfig, rect):
            if info['class'] is Axes3D:
                ax = subfig.add_axes(rect, projection='3d')
                self.ax3D = ax
                self.render_3D(render_all=True)
                ax.view_init(
                    elev=info['ax.elev'],
                    azim=info['ax.azim'],
                    roll=info['ax.roll'],
                    vertical_axis=info['ax._vertical_axis'],
                )
                # matplotlib source code shows that proj_type='orth' is just
                # focal_length=np.inf, so just always pass 'persp' to solve the
                # issue
                ax.set_proj_type(
                    proj_type='persp',
                    focal_length=info['ax._focal_length'],
                )
            else:
                ax = subfig.add_axes(rect)
                self.ax2D = ax
                self.render_2D(render_all=True)
            return ax
        info_main = get_info(self.axd['main'])
        info_alt = get_info(self.axd['alt'])
        self.subfigs0[0].delaxes(self.axd['main'])
        self.subfigs1[1].delaxes(self.axd['alt'])
        self.axd['main'] = consume_info(info_alt, self.subfigs0[0], self.rect_main)
        self.axd['alt'] = consume_info(info_main, self.subfigs1[1], self.rect_alt)
        self.fig.canvas.draw_idle()

    def toggle_reflections(self, event):
        self.on_data_changed(*range(self.num_ops))

    def on_change_reflection_alpha(self, event):
        do_show_reflections = self.translation_tables['reflections-button'][
            self.registry['reflections-button'].get_state()
        ]
        if do_show_reflections:
            self.on_data_changed(*range(self.num_ops))

    def toggle_op_mode(self, op_id, event):
        self.on_data_changed(op_id)

    def update_lamb(self, op_id, lamb_id, val):
        self.on_data_changed(op_id)

    def update_axang(self, op_id, axang_id, val):
        self.on_data_changed(op_id)

    def get_RealSymmetricOperator3(self, op_id):
        operator = RealSymmetricOperator3(
            np.array([x.val for x in self.registry['lamb-slider'][op_id]]),
            axis_angle_to_matrix(*[x.val for x in self.registry['axang-slider'][op_id]])
        )
        return operator

    def render_2D(self, render_all=False):
        pass #TODO

    def render_3D(self, render_all=False):
        # It's easier to just remember how to plot data then deal with figuring
        # out how to recover it directly from the Axes3D, so we do this
        did_render = False
        for op_id, op_data in enumerate(self.render_data['3D']['op']):
            if op_data['stale'] or render_all:
                op_data['stale'] = False
                did_render = True
                self.ax3D.autoscale()
                for artist in op_data['artists']:
                    artist.remove()
                op_data['artists'] = []
                mode = self.translation_tables['mode-button'][
                    self.registry['mode-button'][op_id].get_state()
                ]
                op_data['artists'].append(self.ax3D.plot_trisurf(
                    op_data[mode]['x'],
                    op_data[mode]['y'],
                    self.render_data['3D']['domain']['triangles'],
                    op_data[mode]['z'],
                    color=self.ops_colorwheel[op_id],
                ))
                do_show_reflections = self.translation_tables['reflections-button'][
                    self.registry['reflections-button'].get_state()
                ]
                if do_show_reflections:
                    for i, mode_data in enumerate(op_data[mode]['reflections']):
                        triangles = self.render_data['3D']['domain']['triangles']
                        if i in {0,1,3,6}:
                            triangles = triangles[:, ::-1]
                        op_data['artists'].append(self.ax3D.plot_trisurf(
                            mode_data['x'],
                            mode_data['y'],
                            triangles,
                            mode_data['z'],
                            color=(*self.ops_colorwheel[op_id], self.registry['reflection-alpha'].val),
                        ))
        if did_render:
            set_aspect_equal_3d(self.ax3D)

    def init_data(self):
        self.render_data = {
            '3D': {
                'op': [{
                    'artists': [],
                    'lemniscate': {
                        'reflections': [{} for _ in range(7)],
                    },
                    'pos_shell': {
                        'reflections': [{} for _ in range(7)],
                    },
                    'neg_shell': {
                        'reflections': [{} for _ in range(7)],
                    },
                } for _ in range(self.num_ops)],
            },
            '2D': {
            }
        }
        vectors, triangles = make_simplex_triangles(granularity=5)
        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
        self.render_data['3D']['domain'] = {
            'vectors': vectors,
            'triangles': triangles,
        }

    def on_data_changed(self, *op_id_arr):
        for op_id in op_id_arr:
            self.render_data['3D']['op'][op_id]['stale'] = True
            vectors = self.render_data['3D']['domain']['vectors']
            operator = self.get_RealSymmetricOperator3(op_id)
            r = operator.get_basisless_rayleigh(vectors)
            rotd_vecs = np.squeeze(operator.eigenvectors @ vectors[..., None], -1)
            xyz = r[..., None] * rotd_vecs
            self.render_data['3D']['op'][op_id]['lemniscate'].update({
                'x': xyz[:, 0],
                'y': xyz[:, 1],
                'z': xyz[:, 2],
            })
            xyz = rotd_vecs.copy()
            xyz[0. > r] = np.nan
            self.render_data['3D']['op'][op_id]['pos_shell'].update({
                'x': xyz[:, 0],
                'y': xyz[:, 1],
                'z': xyz[:, 2],
            })
            xyz = rotd_vecs.copy()
            xyz[0. < r] = np.nan
            self.render_data['3D']['op'][op_id]['neg_shell'].update({
                'x': xyz[:, 0],
                'y': xyz[:, 1],
                'z': xyz[:, 2],
            })
            for i in range(7): #enumerate(self.render_data['3D']['op'][op_id]['lemniscate']['reflections']):
                perm_vectors = vectors.copy()
                for j in range(3):
                    if 0 != ((i+1) & (1<<j)):
                        perm_vectors[..., j] *= -1
                rotd_vecs = np.squeeze(operator.eigenvectors @ perm_vectors[..., None], -1)
                xyz = r[..., None] * rotd_vecs
                self.render_data['3D']['op'][op_id]['lemniscate']['reflections'][i].update({
                    'x': xyz[:, 0],
                    'y': xyz[:, 1],
                    'z': xyz[:, 2],
                })
                xyz = rotd_vecs.copy()
                xyz[0. > r] = np.nan
                self.render_data['3D']['op'][op_id]['pos_shell']['reflections'][i].update({
                    'x': xyz[:, 0],
                    'y': xyz[:, 1],
                    'z': xyz[:, 2],
                })
                xyz = rotd_vecs.copy()
                xyz[0. < r] = np.nan
                self.render_data['3D']['op'][op_id]['neg_shell']['reflections'][i].update({
                    'x': xyz[:, 0],
                    'y': xyz[:, 1],
                    'z': xyz[:, 2],
                })

        self.render_3D()
        self.render_2D()

    def get_info_message(self):
        msg_arr = [
            '''
                This interactive renders real symmetric operators in 3D. There
                are 3 visualization modes, togglable per operator with the
                button in the right panel:
            ''', '''
                * Lemniscate: The surface generated by {(x^T A x) x : ||x||_2 = 1}
            ''', '''
                * +Shell: The surface generated by {x : ||x||_2 = 1 AND x^T A x > 0}
            ''', '''
                * -Shell: The surface generated by {x : ||x||_2 = 1 AND x^T A x < 0}
            ''', '''
            ''', '''
                Since these surfaces are all symmetric about their eigen-axes,
                you can toggle whether or not to render all 7 reflections (it
                can sometimes be useful not to render them when negative
                eigenvalues are present). These reflections are shown with an
                alpha channel, controllable via the alpha slider in the top
                right.
            ''', '''
            ''', '''
                The eigenvalues for each operator are controllable
                independently with the 3 sliders in the right panel. The
                eigenbasis is encoded as a member of SO(3) and controllable via
                an axis-angle representation with the 4 sliders in the right
                panel.
            ''',
        ]
        msg_arr = [textwrap.dedent(x) for x in msg_arr]
        msg_arr = [textwrap.fill(x, width=120) for x in msg_arr]
        return '\n'.join(msg_arr)

    def run(self):
        print(self.get_info_message())
        plt.show()


App().run()

#axamp = fig.add([0.25, .03, 0.50, 0.02])
## Slider
#samp = Slider(axamp, 'Amp', 0, 1, valinit=initial_amp)
#
#def update(val):
#    # amp is the current value of the slider
#    amp = samp.val
#    # update curve
#    l.set_ydata(amp*np.sin(t))
#    # redraw canvas while idle
#    fig.canvas.draw_idle()
#
## call update function on slider value change
#samp.on_changed(update)
#
#plt.show()
