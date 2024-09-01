
import textwrap
import functools
from dataclasses import dataclass, field

import numpy as np
import matplotlib as mpl
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
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    triangles = np.array(triangle_arr, dtype=np.int32)
    return vectors, triangles

def axis_angle_to_matrix(theta, v0, v1, v2):
    assert isinstance(theta, float)
    assert isinstance(v0, float)
    assert isinstance(v1, float)
    assert isinstance(v2, float)
    theta *= np.pi
    v = np.array([v0, v1, v2])
    del v0, v1, v2
    norm_v = np.linalg.norm(v)
    if 0. == norm_v:
        return np.eye(3)
    v /= norm_v
    return (
        np.outer(v, v) +
        (np.cos(theta) * (np.eye(3) - np.outer(v, v))) +
        (np.sin(theta) * np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]))
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
                'hiding',
            ),
            'reflections-button': (
                True,
                False,
            ),
            'axislines-button': (
                True,
                False,
            ),
            'slice-button': (
                True,
                False,
            ),
        }
        self._refs_keep_alive = []
        self.registry = {
            'mode-button': [None for _ in range(self.num_ops)],
            'lamb-slider': [[] for _ in range(self.num_ops)],
            'axang-slider': [[] for _ in range(self.num_ops)],
            'reflections-button': None,
            'axislines-button': None,
            'slice-button': None,
            'reflection-alpha': None,
            'granularity-slider': None,
            'slice-slider': [],
        }
        margin_fig_x = 0.01
        margin_fig_y = 0.02
        fontsize = 8
        fig = plt.figure()
        self.fig = fig
        subfigs0_width_ratios = [8, 1, 4]
        subfigs0 = fig.subfigures(
            1, 3,
            wspace=0,
            hspace=0,
            width_ratios=subfigs0_width_ratios,
        )
        self.subfigs0 = subfigs0
        axd = {}
        subfigs01_height_ratios = [1, 1, 1, 1, 7, 7, 7]
        axd.update(subfigs0[1].subplot_mosaic(
            [
                ['swap-views', 'swap-views'],
                ['toggle-reflections', 'toggle-reflections'],
                ['toggle-axislines', 'toggle-axislines'],
                ['toggle-slice', 'toggle-slice'],
                ['reflection-alpha', 'granularity'],
                ['slice-angle', 'slice-axis0'],
                ['slice-axis1', 'slice-axis2'],
            ],
            height_ratios=subfigs01_height_ratios,
        ))
        subfigs0[1].subplots_adjust(
            left=0.01, right=0.97,
            #bottom=margin_fig_y, # Have to adjust for slider text
            bottom=margin_fig_y+0.02,
            top=1.-margin_fig_y,
            wspace=0.1, hspace=0.4,
        )
        subfigs1_height_ratios = [10, 7]
        subfigs1 = subfigs0[2].subfigures(
            2, 1,
            wspace=0,
            hspace=0,
            height_ratios=subfigs1_height_ratios,
        )
        self.subfigs1 = subfigs1
        tmp = subfigs1[0].subplots(4*self.num_ops, 2)
        subfigs1[0].subplots_adjust(
            left=0.01,
            bottom=0.01,
            right=1.-((sum(subfigs0_width_ratios)/subfigs0_width_ratios[2])*margin_fig_x),
            top=1.-((sum(subfigs1_height_ratios)/subfigs1_height_ratios[0])*margin_fig_y),
            wspace=0.1, hspace=0.2,
        )
        for op_id in range(self.num_ops):
            axd[f'op{op_id}mode'] = tmp[4*op_id][0]
            axd[f'op{op_id}angle'] = tmp[4*op_id][1]
            for dim in range(3):
                axd[f'op{op_id}lamb{dim}'] = tmp[(4*op_id)+1+dim][0]
                axd[f'op{op_id}axis{dim}'] = tmp[(4*op_id)+1+dim][1]

        left = (sum(subfigs0_width_ratios)/subfigs0_width_ratios[0])*margin_fig_x
        self.rect_main = (
            left, # left
            margin_fig_y, # bottom
            0.99-left, # width
            1-(2*margin_fig_y), # height
        )
        axd['main'] = subfigs0[0].add_axes(self.rect_main, projection='3d')
        left = 0.01
        right_pad = (sum(subfigs0_width_ratios)/subfigs0_width_ratios[2])*margin_fig_x
        bottom = (sum(subfigs1_height_ratios)/subfigs1_height_ratios[1])*margin_fig_y
        self.rect_alt = (
            left, # left
            bottom, # bottom
            1.-(left+right_pad), # width
            0.99-bottom, # height
        )
        axd['alt'] = subfigs1[1].add_axes(self.rect_alt)
        self.axd = axd

        # General
        tmp = self._keep_alive(widgets.Button(axd['swap-views'], 'Swap Views'))
        tmp.label.set_fontsize(fontsize)
        tmp.on_clicked(self.swap_views)
        tmp = self._keep_alive(StateButton(axd['toggle-reflections'], (
            'Hide Refls',
            'Show Refls',
        )))
        self.registry['reflections-button'] = tmp
        tmp.label.set_fontsize(fontsize)
        tmp.on_clicked(self.toggle_reflections)
        tmp = self._keep_alive(StateButton(axd['toggle-axislines'], (
            'Hide Eig-Axes',
            'Show Eig-Axes',
        )))
        self.registry['axislines-button'] = tmp
        tmp.label.set_fontsize(fontsize)
        tmp.on_clicked(self.toggle_axislines)
        tmp = self._keep_alive(StateButton(axd['toggle-slice'], (
            'Hide Slice',
            'Show Slice',
        )))
        self.registry['slice-button'] = tmp
        tmp.label.set_fontsize(fontsize)
        tmp.on_clicked(self.toggle_slice)
        tmp = self._keep_alive(widgets.Slider(
            axd['reflection-alpha'], r'$\alpha$',
            0., 1., valinit=0.5, valstep=0.05, orientation='vertical',
        ))
        self.registry['reflection-alpha'] = tmp
        tmp.label.set_fontsize(fontsize)
        tmp.valtext.set_fontsize(fontsize)
        tmp.on_changed(self.on_change_reflection_alpha)
        tmp = self._keep_alive(widgets.Slider(
            axd['granularity'], 'Res',
            3, 7, valinit=4, valstep=1, orientation='vertical',
        ))
        self.registry['granularity-slider'] = tmp
        tmp.label.set_fontsize(fontsize)
        tmp.valtext.set_fontsize(fontsize)
        tmp.on_changed(self.on_change_granularity)
        tmp = self._keep_alive(widgets.Slider(
            axd['slice-angle'], rf'$\theta(S)/\pi$',
            -1., 1., valinit=0., valstep=0.05, orientation='vertical',
        ))
        self.registry['slice-slider'].append(tmp)
        tmp.label.set_fontsize(fontsize)
        tmp.valtext.set_fontsize(fontsize)
        tmp.on_changed(functools.partial(self.update_slice, dim))
        for dim in range(3):
            tmp = self._keep_alive(widgets.Slider(
                axd[f'slice-axis{dim}'], rf'Axis$_{dim}(S)$',
                -1., 1., valinit=(1. if 0==dim else 0.), valstep=0.05, orientation='vertical',
            ))
            self.registry['slice-slider'].append(tmp)
            tmp.label.set_fontsize(fontsize)
            tmp.valtext.set_fontsize(fontsize)
            tmp.on_changed(functools.partial(self.update_slice, dim))

        # Per-Op
        color_arr = ('r', 'g', 'b')
        for op_id in range(self.num_ops):
            tmp = self._keep_alive(StateButton(axd[f'op{op_id}mode'], (
                {'text': f'Showing $A_{op_id}$ as Lemniscate', 'color': color_arr[op_id]},
                {'text': f'Showing $A_{op_id}$ as +Shell', 'color': color_arr[op_id]},
                {'text': f'Showing $A_{op_id}$ as -Shell', 'color': color_arr[op_id]},
                {'text': f'Hiding $A_{op_id}$', 'color': 'k'},
            )).set_state(0 if (0==op_id) else 3))
            self.registry['mode-button'][op_id] = tmp
            tmp.label.set_fontsize(fontsize)
            tmp.on_clicked(functools.partial(self.toggle_op_mode, op_id))
            for dim in range(3):
                tmp = self._keep_alive(SliderStyle0(
                    fontsize, axd[f'op{op_id}lamb{dim}'], rf'$\lambda_{dim}(A_{op_id})$',
                    -9., 9., valinit=1., valstep=0.1,
                ))
                self.registry['lamb-slider'][op_id].append(tmp)
                tmp.on_changed(functools.partial(self.update_lamb, op_id, dim))
            tmp = self._keep_alive(SliderStyle0(
                fontsize, axd[f'op{op_id}angle'], rf'$\theta(A_{op_id})/\pi$',
                -1., 1., valinit=0., valstep=0.05,
            ))
            self.registry['axang-slider'][op_id].append(tmp)
            tmp.on_changed(functools.partial(self.update_axang, op_id, dim))
            for dim in range(3):
                tmp = self._keep_alive(SliderStyle0(
                    fontsize, axd[f'op{op_id}axis{dim}'], rf'Axis$_{dim}(A_{op_id})$',
                    -1., 1., valinit=(1. if 0==dim else 0.), valstep=0.05,
                ))
                self.registry['axang-slider'][op_id].append(tmp)
                tmp.on_changed(functools.partial(self.update_axang, op_id, dim))


        # Initial Cofiguration
        self.ax3D = axd['main']
        self.ax3D.set_xticks([])
        self.ax3D.set_yticks([])
        self.ax3D.set_zticks([])
        self.ax2D = axd['alt']
        self.ax2D.set_xticks([])
        self.ax2D.set_yticks([])

        self.init_data()

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
                info['ax.xticks'] = ax.get_xticks()
                info['ax.yticks'] = ax.get_yticks()
                info['ax.zticks'] = ax.get_zticks()
                info['ax.xlim'] = ax.get_xlim()
                info['ax.ylim'] = ax.get_ylim()
                info['ax.zlim'] = ax.get_zlim()
                # It's easier to just remember how to plot data then deal with
                # figuring out how to recover it directly from the Axes3D, so
                # we do that with self.render_data
            else:
                info['class'] = Axes
                info['ax.xticks'] = ax.get_xticks()
                info['ax.yticks'] = ax.get_yticks()
                info['ax.xlim'] = ax.get_xlim()
                info['ax.ylim'] = ax.get_ylim()
                info['ax.frame_on'] = ax.get_frame_on()
            return info
        def consume_info_pre(info, subfig, rect):
            if info['class'] is Axes3D:
                ax = subfig.add_axes(rect, projection='3d')
                self.ax3D = ax
            else:
                ax = subfig.add_axes(rect)
                self.ax2D = ax
            return ax
        def consume_info_post(ax, info):
            if info['class'] is Axes3D:
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
                ax.set_xticks(info['ax.xticks'])
                ax.set_yticks(info['ax.yticks'])
                ax.set_zticks(info['ax.zticks'])
                ax.set_xlim(info['ax.xlim'])
                ax.set_ylim(info['ax.ylim'])
                ax.set_zlim(info['ax.zlim'])
            else:
                ax.set_xticks(info['ax.xticks'])
                ax.set_yticks(info['ax.yticks'])
                ax.set_xlim(info['ax.xlim'])
                ax.set_ylim(info['ax.ylim'])
                ax.set_frame_on(info['ax.frame_on'])
        info_main = get_info(self.axd['main'])
        info_alt = get_info(self.axd['alt'])
        self.subfigs0[0].delaxes(self.axd['main'])
        self.subfigs1[1].delaxes(self.axd['alt'])
        self.axd['main'] = consume_info_pre(info_alt, self.subfigs0[0], self.rect_main)
        self.axd['alt'] = consume_info_pre(info_main, self.subfigs1[1], self.rect_alt)
        self.render(render_all=True)
        consume_info_post(self.axd['main'], info_alt)
        consume_info_post(self.axd['alt'], info_main)
        self.fig.canvas.draw_idle()

    def toggle_reflections(self, event):
        self.on_data_changed(*range(self.num_ops))

    def toggle_axislines(self, event):
        self.render_data['axislines']['stale'] = True
        self.render()

    def toggle_slice(self, event):
        self.render_data['slice']['stale'] = True
        self.render()

    def on_change_reflection_alpha(self, val):
        do_show_reflections = self.translation_tables['reflections-button'][
            self.registry['reflections-button'].get_state()
        ]
        if do_show_reflections:
            for op_id in range(self.num_ops):
                self.render_data['op'][op_id]['3D']['stale'] = True
                self.render_data['op'][op_id]['2D']['stale'] = True
            self.render()

    def on_change_granularity(self, val):
        self.on_data_changed(*range(self.num_ops), only_dimlty='3D')

    def update_slice(self, slice_axang_id, val):
        self.on_slice_changed()

    def toggle_op_mode(self, op_id, event):
        self.render_data['axislines']['stale'] = True
        self.on_data_changed(op_id)
        self.on_slice_changed(only_if_r_changed=True)

    def update_lamb(self, op_id, lamb_id, val):
        self.render_data['axislines']['stale'] = True
        self.on_data_changed(op_id)
        self.on_slice_changed(only_if_r_changed=True)

    def update_axang(self, op_id, axang_id, val):
        self.render_data['axislines']['stale'] = True
        self.on_data_changed(op_id)

    def get_RealSymmetricOperator3(self, op_id):
        operator = RealSymmetricOperator3(
            np.array([x.val for x in self.registry['lamb-slider'][op_id]]),
            axis_angle_to_matrix(*[x.val for x in self.registry['axang-slider'][op_id]])
        )
        return operator

    def render(self, render_all=False):
        ALPHA_MAIN = 0.95
        # It's easier to just remember how to plot data then deal with figuring
        # out how to recover it directly from the Axes3D, so we do this
        did_render_3D = False
        did_render_2D = False
        if render_all:
            self.ax3D.scatter(
                [0], [0], [0],
                color=(0., 0., 0., 0.4),
                marker='.'
            )
            self.ax2D.scatter(
                [0], [0],
                color=(0., 0., 0., 0.4),
                marker='.'
            )
        for op_id, op_data in enumerate(self.render_data['op']):
            if render_all:
                for dimlty in ('3D', '2D'):
                    op_data[dimlty]['stale'] = True
            if any(op_data[x]['stale'] for x in ('3D', '2D')):
                stale_3D = op_data['3D']['stale']
                stale_2D = op_data['2D']['stale']
                op_data['3D']['stale'] = False
                op_data['2D']['stale'] = False
                mode = self.translation_tables['mode-button'][
                    self.registry['mode-button'][op_id].get_state()
                ]
                if stale_3D:
                    for artist in op_data['3D']['artists']:
                        artist.remove()
                    op_data['3D']['artists'] = []
                if stale_2D:
                    for artist in op_data['2D']['artists']:
                        artist.remove()
                    op_data['2D']['artists'] = []
                if 'hiding' == mode:
                    continue
                if stale_3D:
                    #self.ax3D.autoscale()
                    did_render_3D = True
                    op_data['3D']['artists'].append(self.ax3D.plot_trisurf(
                        op_data['3D'][mode]['x'],
                        op_data['3D'][mode]['y'],
                        self._get_domain('3D')['triangles'],
                        op_data['3D'][mode]['z'],
                        color=(*self.ops_colorwheel[op_id], ALPHA_MAIN),
                    ))
                if stale_2D:
                    did_render_2D = True
                    op_data['2D']['artists'].append(self.ax2D.plot(
                        op_data['2D'][mode]['main']['x'],
                        op_data['2D'][mode]['main']['y'],
                        color=(*self.ops_colorwheel[op_id], ALPHA_MAIN),
                    )[0])
                    tmp = self.render_data['slice']['r'] * np.array([[1, 1], [0, 1]])
                    op_data['2D']['artists'].append(self.ax2D.scatter(
                        tmp[0],
                        tmp[1],
                        color=(0., 0., 0., 0.4),
                        marker='x',
                    ))
                do_show_reflections = self.translation_tables['reflections-button'][
                    self.registry['reflections-button'].get_state()
                ]
                if do_show_reflections:
                    if stale_3D:
                        for i, mode_data in enumerate(op_data['3D'][mode]['reflections']):
                            triangles = self._get_domain('3D')['triangles']
                            if i in {0,1,3,6}:
                                triangles = triangles[:, ::-1]
                            op_data['3D']['artists'].append(self.ax3D.plot_trisurf(
                                mode_data['x'],
                                mode_data['y'],
                                triangles,
                                mode_data['z'],
                                color=(*self.ops_colorwheel[op_id], self.registry['reflection-alpha'].val),
                            ))
                    if stale_2D:
                        op_data['2D']['artists'].append(self.ax2D.plot(
                            op_data['2D'][mode]['refl']['x'],
                            op_data['2D'][mode]['refl']['y'],
                            color=(*self.ops_colorwheel[op_id], self.registry['reflection-alpha'].val),
                        )[0])
        axislines_data = self.render_data['axislines']
        if axislines_data['stale'] or render_all:
            axislines_data['stale'] = False
            for artist in axislines_data['3D']['artists']:
                artist.remove()
            axislines_data['3D']['artists'] = []
            for artist in axislines_data['2D']['artists']:
                artist.remove()
            axislines_data['2D']['artists'] = []
            do_show_axislines = self.translation_tables['axislines-button'][
                self.registry['axislines-button'].get_state()
            ]
            if do_show_axislines:
                for op_id in range(self.num_ops):
                    mode = self.translation_tables['mode-button'][
                        self.registry['mode-button'][op_id].get_state()
                    ]
                    if 'hiding' == mode:
                        continue
                    #self.ax3D.autoscale()
                    operator = self.get_RealSymmetricOperator3(op_id)
                    did_render_3D = True
                    pm1 = np.array([-1., 1])
                    for i in range(3):
                        r = operator.eigenvalues[i] if 'lemniscate'==mode else 1.
                        axislines_data['3D']['artists'].append(self.ax3D.plot(
                            1.1 * r * operator.eigenvectors[0, i] * pm1,
                            1.1 * r * operator.eigenvectors[1, i] * pm1,
                            1.1 * r * operator.eigenvectors[2, i] * pm1,
                            color=(*self.ops_colorwheel[op_id], ALPHA_MAIN),
                        )[0])
                    did_render_2D = True
                    proj_eigvecs = self.render_data['slice']['basis'].T @ operator.eigenvectors
                    for i in range(3):
                        r = operator.eigenvalues[i] if 'lemniscate'==mode else 1.
                        axislines_data['2D']['artists'].append(self.ax2D.plot(
                            r * proj_eigvecs[0, i] * pm1,
                            r * proj_eigvecs[1, i] * pm1,
                            color=(*self.ops_colorwheel[op_id], ALPHA_MAIN),
                        )[0])
        slice_data = self.render_data['slice']
        if slice_data['stale'] or render_all:
            slice_data['stale'] = False
            for artist in slice_data['artists']:
                artist.remove()
            slice_data['artists'] = []
            do_show_slice = self.translation_tables['slice-button'][
                self.registry['slice-button'].get_state()
            ]
            if do_show_slice:
                #self.ax3D.autoscale()
                did_render_3D = True
                slice_data['artists'].append(self.ax3D.plot_surface(
                    slice_data['x'],
                    slice_data['y'],
                    slice_data['z'],
                    color=(1., 1., 1., 0.4),
                ))
                tmp = slice_data['r'] * np.cumsum(slice_data['basis'], axis=-1)
                slice_data['artists'].append(self.ax3D.scatter(
                    tmp[0],
                    tmp[1],
                    tmp[2],
                    color=(0., 0., 0., 0.4),
                    marker='x',
                ))
        if did_render_3D:
            set_aspect_equal_3d(self.ax3D)
        if did_render_2D:
            self.ax2D.set_aspect('equal', 'box')
            r = self.render_data['slice']['r']
            self.ax2D.set(xlim=(-r, r), ylim=(-r, r))

    def _get_domain(self, dimlty):
        assert dimlty in ('3D', '2D')
        if '3D' == dimlty:
            domain_dict = self.render_data['domain'][dimlty]
            granularity = self.registry['granularity-slider'].val
            if granularity not in domain_dict:
                vectors, triangles = make_simplex_triangles(granularity=granularity)
                domain_dict[granularity] = {
                    'vectors': vectors,
                    'triangles': triangles,
                }
            return domain_dict[granularity]
        else:
            return self.render_data['domain'][dimlty]

    def init_data(self):
        num_grid = 256
        tmp = np.arange(num_grid) / num_grid
        domain_2D = np.stack((
            np.cos(2*np.pi*tmp),
            np.sin(2*np.pi*tmp),
        ), axis=-1)
        num_grid = 32
        tmp = np.linspace(-1, 1, num_grid)
        slice_domain_vecs = np.stack((
            *np.meshgrid(tmp, tmp),
            np.zeros((num_grid, num_grid))
        ), axis=-1)
        self.render_data = {
            'domain': {
                '3D': {},
                '2D': domain_2D,
            },
            'axislines': {
                'stale': False,
                '3D': {
                    'artists': [],
                },
                '2D': {
                    'artists': [],
                },
            },
            'slice': {
                'domain-vectors': slice_domain_vecs,
                'stale': False,
                'artists': [],
                'r': None,
                'basis': None,
            },
            'op': [{
                '3D': {
                    'stale': False,
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
                },
                '2D': {
                    'stale': False,
                    'artists': [],
                    'lemniscate': {
                        'main': {},
                        'refl': {},
                    },
                    'pos_shell': {
                        'main': {},
                        'refl': {},
                    },
                    'neg_shell': {
                        'main': {},
                        'refl': {},
                    },
                },
            } for _ in range(self.num_ops)],
        }
        self._get_domain('3D')
        self._get_domain('2D')
        self.on_slice_changed()
        self.on_data_changed(*range(self.num_ops))
        self.render(render_all=True)

    def on_data_changed(self, *op_id_arr, only_dimlty=None):
        assert only_dimlty in (None, '3D', '2D')
        for op_id in op_id_arr:
            operator = self.get_RealSymmetricOperator3(op_id)
            if '2D' != only_dimlty:
                self.render_data['op'][op_id]['3D']['stale'] = True
                vectors = self._get_domain('3D')['vectors']
                r = operator.get_basisless_rayleigh(vectors)
                rotd_vecs = np.squeeze(operator.eigenvectors @ vectors[..., None], -1)
                xyz = r[..., None] * rotd_vecs
                self.render_data['op'][op_id]['3D']['lemniscate'].update({
                    'x': xyz[:, 0],
                    'y': xyz[:, 1],
                    'z': xyz[:, 2],
                })
                xyz = rotd_vecs.copy()
                xyz[0. > r] = np.nan
                self.render_data['op'][op_id]['3D']['pos_shell'].update({
                    'x': xyz[:, 0],
                    'y': xyz[:, 1],
                    'z': xyz[:, 2],
                })
                xyz = rotd_vecs.copy()
                xyz[0. < r] = np.nan
                self.render_data['op'][op_id]['3D']['neg_shell'].update({
                    'x': xyz[:, 0],
                    'y': xyz[:, 1],
                    'z': xyz[:, 2],
                })
                for i in range(7):
                    perm_vectors = vectors.copy()
                    for j in range(3):
                        if 0 != ((i+1) & (1<<j)):
                            perm_vectors[..., j] *= -1
                    rotd_vecs = np.squeeze(operator.eigenvectors @ perm_vectors[..., None], -1)
                    xyz = r[..., None] * rotd_vecs
                    self.render_data['op'][op_id]['3D']['lemniscate']['reflections'][i].update({
                        'x': xyz[:, 0],
                        'y': xyz[:, 1],
                        'z': xyz[:, 2],
                    })
                    xyz = rotd_vecs.copy()
                    xyz[0. > r] = np.nan
                    self.render_data['op'][op_id]['3D']['pos_shell']['reflections'][i].update({
                        'x': xyz[:, 0],
                        'y': xyz[:, 1],
                        'z': xyz[:, 2],
                    })
                    xyz = rotd_vecs.copy()
                    xyz[0. < r] = np.nan
                    self.render_data['op'][op_id]['3D']['neg_shell']['reflections'][i].update({
                        'x': xyz[:, 0],
                        'y': xyz[:, 1],
                        'z': xyz[:, 2],
                    })
            if '3D' != only_dimlty:
                self.render_data['op'][op_id]['2D']['stale'] = True
                vecs_2D = self._get_domain('2D')
                basis = self.render_data['slice']['basis']
                vecs_3D = np.squeeze(basis @ vecs_2D[..., None], -1)
                r = operator.get_rayleigh(vecs_3D)
                tmp = np.squeeze(operator.eigenvectors.T @ vecs_3D[..., None], -1)
                refl_mask = np.any(0. > tmp, axis=-1)
                ind = np.argwhere(np.diff(refl_mask))
                ind = 1 + np.squeeze(ind, 1)
                assert 1 == ind.ndim
                # Might not be true due to numerical instability? Use -1 index
                # below for safety
                #assert 2 >= ind.size
                if 0 == ind.size:
                    vecs_2D_main = np.concatenate((vecs_2D, vecs_2D[[0]]), axis=0)
                    vecs_2D_refl = np.zeros((0, 2))
                    r_main = np.concatenate((r, r[[0]]), axis=0)
                    r_refl = np.zeros((0,))
                elif 1 == ind.size:
                    vecs_2D_main = vecs_2D[:ind[0]+1]
                    vecs_2D_refl = np.concatenate((vecs_2D[ind[0]:], vecs_2D[[0]]), axis=0)
                    r_main = r[:ind[0]+1]
                    r_refl = np.concatenate((r[ind[0]:], r[[0]]), axis=0)
                else:
                    vecs_2D_main = np.concatenate((vecs_2D[ind[-1]:], vecs_2D[:ind[0]+1]), axis=0)
                    vecs_2D_refl = vecs_2D[ind[0]:ind[-1]+1]
                    r_main = np.concatenate((r[ind[-1]:], r[:ind[0]+1]), axis=0)
                    r_refl = r[ind[0]:ind[-1]+1]
                if refl_mask[0]:
                    tmp = vecs_2D_main
                    vecs_2D_main = vecs_2D_refl
                    vecs_2D_refl = tmp
                    tmp = r_main
                    r_main = r_refl
                    r_refl = tmp
                xy_main = r_main[:, None] * vecs_2D_main
                xy_refl = r_refl[:, None] * vecs_2D_refl
                self.render_data['op'][op_id]['2D']['lemniscate']['main'].update({
                    'x': xy_main[:, 0],
                    'y': xy_main[:, 1],
                })
                self.render_data['op'][op_id]['2D']['lemniscate']['refl'].update({
                    'x': xy_refl[:, 0],
                    'y': xy_refl[:, 1],
                })
                xy_main = vecs_2D_main.copy()
                xy_main[0. > r_main] = np.nan
                xy_refl = vecs_2D_refl.copy()
                xy_refl[0. > r_refl] = np.nan
                self.render_data['op'][op_id]['2D']['pos_shell']['main'].update({
                    'x': xy_main[:, 0],
                    'y': xy_main[:, 1],
                })
                self.render_data['op'][op_id]['2D']['pos_shell']['refl'].update({
                    'x': xy_refl[:, 0],
                    'y': xy_refl[:, 1],
                })
                xy_main = vecs_2D_main.copy()
                xy_main[0. < r_main] = np.nan
                xy_refl = vecs_2D_refl.copy()
                xy_refl[0. < r_refl] = np.nan
                self.render_data['op'][op_id]['2D']['neg_shell']['main'].update({
                    'x': xy_main[:, 0],
                    'y': xy_main[:, 1],
                })
                self.render_data['op'][op_id]['2D']['neg_shell']['refl'].update({
                    'x': xy_refl[:, 0],
                    'y': xy_refl[:, 1],
                })
        self.render()

    def _get_ideal_slice_r(self):
        r = -1.
        for op_id, lamb_slider_arr in enumerate(self.registry['lamb-slider']):
            mode = self.translation_tables['mode-button'][
                self.registry['mode-button'][op_id].get_state()
            ]
            if 'lemniscate' == mode:
                r = max([r] + [abs(x.val) for x in lamb_slider_arr])
        if -1. == r:
            r = 1.
        r *= 1.1
        return r

    def on_slice_changed(self, only_if_r_changed=False):
        r = self._get_ideal_slice_r()
        if r != self.render_data['slice']['r']:
            self.render_data['slice']['r'] = r
        elif only_if_r_changed:
            return
        self.render_data['slice']['stale'] = True
        self.render_data['axislines']['stale'] = True
        vectors = self.render_data['slice']['domain-vectors']
        basis = axis_angle_to_matrix(*[x.val for x in self.registry['slice-slider']])
        self.render_data['slice']['basis'] = basis[:, :2]
        rotd_vecs = np.squeeze(basis @ vectors[..., None], -1)
        xyz = r * rotd_vecs
        self.render_data['slice'].update({
            'x': xyz[..., 0],
            'y': xyz[..., 1],
            'z': xyz[..., 2],
        })
        self.on_data_changed(*range(self.num_ops), only_dimlty='2D')
        self.render()

    def get_info_message(self):
        msg_arr = [
            '''\
                This interactive renders real symmetric operators in 3D. There
                are 3 visualization modes, togglable per operator with the
                button in the right panel:
            ''', '''\
                * \'Lemniscate\': The surface generated by {(x^T A x) x : ||x||_2 = 1}
            ''', '''\
                * \'+Shell\': The surface generated by {x : ||x||_2 = 1 AND x^T A x > 0}
            ''', '''\
                * \'-Shell\': The surface generated by {x : ||x||_2 = 1 AND x^T A x < 0}
            ''', '''
            ''', '''\
                The eigenvalues for each operator are controllable
                independently with the 3 sliders in the right panel. The
                eigenbasis is encoded as a member of SO(3) and controllable via
                an axis-angle representation with the 4 sliders in the right
                panel. The angle theta follows the right-hand rule, rotating
                the entire object along the specified axis. Note that the angle
                slider value as specified by the user defines the angle theta
                divided by pi, i.e. a slider value of 0.5 represents pi/2
                radians. The axis slider values as specified by the user are
                normalized to a unit norm vector prior to all internal logic;
                i.e. an axis of (0.7, 0, 0) is equivalent to an axis of (0.5,
                0, 0), each corresponding to the unit norm axis (1, 0, 0).
            ''', '''
            ''', '''\
                Two visualizations are presented: a 3D total view and a 2D
                sliced plane view. The user may swap the positions of these two
                views with the \'Swap Views\' button. The slice plane from
                which the 2D view is taken is controlled via an axis-angle
                rotation similarly to the eigenbases mentioned above. The slice
                plane is also shown in the 3D view, but may be toggled to hide.
                In both the 3D and 2D views, there are two crosshairs rendered
                along the edge of the slice plane solely as a visual reference
                aid between the two views.
            ''', '''
            ''', '''\
                Since these surfaces are all symmetric about their eigen-axes,
                the user may toggle whether or not to render all 7 reflections
                (it can sometimes be useful not to render them when negative
                eigenvalues are present). These reflections are shown with an
                alpha channel, controllable via the alpha slider.
            ''', '''
            ''', '''\
                The eigen-axes may also be toggled to be shown/hidden. In the 2D
                view, these axes are visualized by their projections onto the
                slice plane (i.e. like a shadow).
            ''', '''
            ''', '''\
                The resolution/granularity of the mesh used for the 3D view may
                be controlled via the \'Res\' slider. A slider value of N means the
                simplex triangle will be subdivided recursively N times for the
                point grid. This slider is provided to allow the user to
                balance between program reaction speed and visual quality.
            ''',
        ]
        msg_arr = [textwrap.dedent(x) for x in msg_arr]
        msg_arr = [textwrap.fill(x, width=120) for x in msg_arr]
        return '\n'.join(msg_arr)

    def run(self):
        print(self.get_info_message())
        plt.show()


if __name__ == '__main__':
    App().run()

