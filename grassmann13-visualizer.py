from util import *

import textwrap
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets

@dataclass
class Grassmannian13:
    point_R3: np.ndarray = field(default_factory=lambda: theta_phi_to_point_R3(0., 0.))
    def verify(self):
        if not (3 == self.point_R3.shape[-1]):
            return False
        if not np.isrealobj(self.point_R3):
            return False
        if 1e-12 < np.max(np.abs(1. - np.linalg.norm(self.point_R3, axis=-1))):
            return False
        return True
    def get_point_R3(self):
        return self.point_R3
    def get_embedding(self):
        P = self.point_R3[..., :, None] * self.point_R3[..., None, :]
        K = 1
        N = 3
        G = ((N / (K * (N - K)))**0.5) * (P - ((K / N) * np.eye(N)))
        return G

def theta_phi_to_point_R3(theta, phi):
    return np.stack((
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)*np.ones_like(phi),
    ), axis=-1)


class App:

    def __init__(self):
        self.translation_tables = {
            'toggle-loop': (
                True,
                False,
            ),
            'toggle-full-mfold': (
                False,
                True,
            ),
        }
        self._refs_keep_alive = []
        self.registry = {
            'embedding-matrix': None,
            'theta-slider': None,
            'phi-slider': None,
            'toggle-loop': None,
            'toggle-full-mfold': None,
        }
        margin_fig_x = 0.01
        margin_fig_y = 0.02
        fontsize = 8
        fig = plt.figure()
        self.fig = fig
        subfigs_height_ratios = [1, 4]
        subfigs = fig.subfigures(
            2, 1,
            wspace=0,
            hspace=0,
            height_ratios=subfigs_height_ratios,
        )
        self.subfigs = subfigs
        axd = {}
        subfigs0_width_ratios = [1, 2, 2, 1]
        axd.update(subfigs[0].subplot_mosaic(
            [
                ['embedding-matrix', 'theta-slider', 'theta-slider', 'point_R3-scene'],
                ['embedding-matrix', 'phi-slider', 'phi-slider', 'point_R3-scene'],
                ['embedding-matrix', 'toggle-loop', 'toggle-full-mfold', 'point_R3-scene'],
            ],
            width_ratios=subfigs0_width_ratios,
            per_subplot_kw={'point_R3-scene': dict(projection='3d')},
        ))
        subfigs[0].subplots_adjust(
            left=margin_fig_x,
            right=1.-margin_fig_x,
            top=1.-((sum(subfigs_height_ratios)/subfigs_height_ratios[0])*margin_fig_y),
        )
        tmp = subfigs[1].subplots(1, 2, subplot_kw=dict(projection='3d'))
        subfigs[1].subplots_adjust(
            left=margin_fig_x,
            right=1.-margin_fig_x,
            bottom=(sum(subfigs_height_ratios)/subfigs_height_ratios[1])*margin_fig_y,
        )
        axd['embed0'] = tmp[0]
        axd['embed1'] = tmp[1]
        self.axd = axd

        # General
        tmp = self._keep_alive(widgets.TextBox(
            axd['embedding-matrix'], '',
            textalignment='center',
        ))
        self.registry['embedding-matrix'] = tmp
        tmp.label.set_fontsize(fontsize)
        tmp = self._keep_alive(widgets.Slider(
            axd['theta-slider'], r'$\theta/\pi$',
            0., 1., valinit=0., valstep=0.05, orientation='horizontal',
        ))
        self.registry['theta-slider'] = tmp
        tmp.label.set_fontsize(fontsize)
        tmp.valtext.set_fontsize(fontsize)
        tmp.on_changed(self.on_sliders_change)
        tmp = self._keep_alive(widgets.Slider(
            axd['phi-slider'], r'$\phi/\pi$',
            0., 2., valinit=0., valstep=0.05, orientation='horizontal',
        ))
        self.registry['phi-slider'] = tmp
        tmp.label.set_fontsize(fontsize)
        tmp.valtext.set_fontsize(fontsize)
        tmp.on_changed(self.on_sliders_change)
        tmp = self._keep_alive(StateButton(axd['toggle-loop'], (
            'Hide Mobius Boundary Loop',
            'Show Mobius Boundary Loop',
        )))
        self.registry['toggle-loop'] = tmp
        tmp.label.set_fontsize(fontsize)
        tmp.on_clicked(self.toggle_loop)
        tmp = self._keep_alive(StateButton(axd['toggle-full-mfold'], (
            'Show Entire Gr(1,3) Manifold',
            'Show Partial Mobius Strip',
        )))
        self.registry['toggle-full-mfold'] = tmp
        tmp.label.set_fontsize(fontsize)
        tmp.on_clicked(self.toggle_full_mfold)

        # Initial Cofiguration
        for ax_name in ['embed0', 'embed1', 'point_R3-scene']:
            axd[ax_name].set_xticks([])
            axd[ax_name].set_yticks([])
            axd[ax_name].set_zticks([])
        for ax_name in ['embed0', 'embed1']:
            axd[ax_name].view_init(
                elev=45,
                azim=45,
                roll=0,
            )
        self.init_data()

    def _keep_alive(self, x):
        self._refs_keep_alive.append(x)
        return x

    def toggle_loop(self, event):
        self.render()

    def toggle_full_mfold(self, event):
        do_show_full_mfold = self.translation_tables['toggle-full-mfold'][
            self.registry['toggle-full-mfold'].get_state()
        ]
        if do_show_full_mfold:
            self.render(render_all=True)
        else:
            self.render()

    def update_embedding_matrix(self):
        matrix = self.get_Grassmannian13().get_embedding()
        prec = 3
        matrix[np.abs(matrix) < 10**(-prec)] = 0.
        with np.printoptions(precision=prec):
            self.registry['embedding-matrix'].set_val(
                'Matrix Embedding:\n' + str(matrix)
            )
            self.axd['embed0'].set_title(str(matrix[range(3), range(3)]))
            self.axd['embed1'].set_title(str(matrix[[1, 0, 0], [2, 2, 1]]))

    def get_Grassmannian13(self):
        theta = np.pi * self.registry['theta-slider'].val
        phi = np.pi * self.registry['phi-slider'].val
        gr13 = Grassmannian13(theta_phi_to_point_R3(theta, phi))
        return gr13

    def render(self, render_all=False):

        def _render_scenes(G_arr, p_arr, artists_key, kw_embed, kw_R3):
            assert isinstance(G_arr, (tuple, list))
            assert isinstance(G_arr[0], (tuple, list))
            assert isinstance(p_arr, (tuple, list))
            if 2 == p_arr[0].ndim-1:
                def plot_func(ax, *args, **kwargs):
                    return ax.plot_surface(*args, **kwargs)
            elif 1 >= p_arr[0].ndim-1:
                def plot_func(ax, *args, **kwargs):
                    return ax.plot3D(*args, **kwargs)[0]
            else:
                raise NotImplementedError
            for artist in self.render_data['artists'][artists_key]:
                artist.remove()
            artists = []
            self.render_data['artists'][artists_key] = artists
            for G in G_arr:
                for i, G_data in enumerate((
                    (G[0][..., 0, 0], G[0][..., 1, 1], G[0][..., 2, 2]),
                    (G[1][..., 1, 2], G[1][..., 0, 2], G[1][..., 0, 1]))
                ):
                    artists.append(plot_func(
                        self.axd[f'embed{i}'],
                        *G_data,
                        **kw_embed,
                    ))
            for p in p_arr:
                artists.append(plot_func(
                    self.axd['point_R3-scene'],
                    p[..., 0], p[..., 1], p[..., 2],
                    **kw_R3,
                ))
                assert 'color' in kw_R3
                kw_R3['color'] = (*kw_R3['color'][:3], kw_R3['color'][3]/2)
                artists.append(plot_func(
                    self.axd['point_R3-scene'],
                    -p[..., 0], -p[..., 1], -p[..., 2],
                    **kw_R3,
                ))

        num_grid = self.render_data['G'].shape[0]-1
        tmp = self.registry['theta-slider'].val
        ind_mobius_loop = int((1 - (2 * abs(0.5 - tmp))) * num_grid)

        do_show_full_mfold = self.translation_tables['toggle-full-mfold'][
            self.registry['toggle-full-mfold'].get_state()
        ]
        if (not do_show_full_mfold) or render_all:
            if not do_show_full_mfold:
                ind = ind_mobius_loop
            else:
                ind = 0
            G0 = self.render_data['G_fourth_embed0'][ind:]
            G1 = self.render_data['G'][ind:]
            p = self.render_data['p'][ind:]
            _render_scenes(
                [[G0, G1]], [p], 'mfold',
                dict(color=(0., 1., 0., 0.3)),
                dict(color=(0., 0., 0., 0.4)),
            )

        do_show_loop = self.translation_tables['toggle-loop'][
            self.registry['toggle-loop'].get_state()
        ]
        if not do_show_loop:
            for artists_key in ('loop', 'fiber'):
                for artist in self.render_data['artists'][artists_key]:
                    artist.remove()
                self.render_data['artists'][artists_key] = []
        else:
            G0 = self.render_data['G_fourth_embed0'][ind_mobius_loop]
            G1 = self.render_data['G'][ind_mobius_loop]
            p = self.render_data['p'][ind_mobius_loop]
            _render_scenes(
                [[G0, G1]], [p], 'loop',
                dict(color=(1., 0., 1., 1.)),
                dict(color=(1., 0., 1., 1.)),
            )
            num_grid = self.render_data['G'].shape[0]-1
            tmp = self.registry['phi-slider'].val
            ind_mobius_fiber = int((0.5 * tmp) * num_grid)
            def reduce_ind_by_sym(ind):
                ind = (ind % (num_grid//2))
                if ind > (num_grid//4):
                    ind = (num_grid//2) - ind
                return ind
            _i = reduce_ind_by_sym(ind_mobius_fiber)
            G0 = self.render_data['G_fourth_embed0'][ind_mobius_loop:, _i]
            G1 = self.render_data['G'][ind_mobius_loop:, ind_mobius_fiber]
            p = self.render_data['p'][ind_mobius_loop:, ind_mobius_fiber]
            ind_refl = (ind_mobius_fiber + (num_grid//2)) % num_grid
            _i = reduce_ind_by_sym(ind_refl)
            G0_refl = self.render_data['G_fourth_embed0'][ind_mobius_loop:, _i]
            G1_refl = self.render_data['G'][ind_mobius_loop:, ind_refl]
            p_refl = self.render_data['p'][ind_mobius_loop:, ind_refl]
            _render_scenes(
                [[G0, G1], [G0_refl, G1_refl]], [p, p_refl], 'fiber',
                dict(color=(0., 0., 1., 0.5), linestyle='--'),
                dict(color=(0., 0., 1., 0.5), linestyle='--'),
            )

        # _render_scenes is not used below because of the baton rendering in
        # point_R3-scene
        gr13 = self.get_Grassmannian13()
        G = gr13.get_embedding()
        p = gr13.get_point_R3()[None, :] * np.array([1., -1])[:, None]
        artists_key = 'point'
        for artist in self.render_data['artists'][artists_key]:
            artist.remove()
        artists = []
        self.render_data['artists'][artists_key] = artists
        for i, G_data in enumerate((
            (G[0, 0], G[1, 1], G[2, 2]),
            (G[1, 2], G[0, 2], G[0, 1]),
        )):
            artists.append(self.axd[f'embed{i}'].plot3D(
                *G_data,
                color=(0., 0., 0., 1.),
                marker='o',
                markersize=10,
            )[0])
        artists.append(self.axd['point_R3-scene'].plot3D(
            p[:, 0], p[:, 1], p[:, 2],
            '-o',
            color=(0., 0., 0., 1.),
        )[0])

        for ax_name in ['embed0', 'embed1', 'point_R3-scene']:
            set_aspect_equal_3d(self.axd[ax_name])

    def on_sliders_change(self, val):
        self.update_embedding_matrix()
        self.render()

    def init_data(self):
        num_grid = 4 * 40 # multiple of slider resolution of 2/0.05 = 40
        theta = np.linspace(0, 0.5*np.pi, num_grid+1)[:, None]
        phi = np.linspace(0, 2*np.pi, num_grid+1)[None, :]
        gr13 = Grassmannian13(theta_phi_to_point_R3(theta, phi))
        G = gr13.get_embedding()
        G_fourth_embed0 = G[:, :(num_grid//4)+1]
        p = gr13.get_point_R3()
        self.render_data = {
            'G': G,
            'G_fourth_embed0': G_fourth_embed0,
            'p': p,
            'artists': {
                'mfold': [],
                'loop': [],
                'fiber': [],
                'point': [],
            }
        }
        self.update_embedding_matrix()
        self.render(render_all=True)

    def get_info_message(self):
        msg_arr = [
            '''\
                This interactive renders the Grassmann(1, 3) manifold
                (equivalent to Real Projective 3-space) as a 2-dimensional
                submanifold of the 5-sphere using the following general
                Grassmann(K, N) embedding (where P is the associated projection
                matrix):
            ''', '''\
                >   G = ((N / (K * (N - K)))**0.5) * (P - (K / N) I)
            ''', '''
            ''', '''\
                This embedding is visualized via a pair of 3-dimensional
                renderings: a left axes for the diagonal elements of G and a
                right axes for the off-diagonal elements of G. The matrix G is
                also displayed in the top left of the interactive. This
                visualization aims to help the user's intuition towards the
                following facts:
            ''', '''\
                * There is a 'natural' embedding of Gr(1, 3) as a submanifold
                of the 5-sphere.
            ''', '''\
                * The topology of Gr(1, 3) is that of a mobius strip whose
                boundary is contracted to a single point. (Note: a fun
                corrolary is that such a topology is only embeddable (without
                self-intersection) in 5-dimensional or higher Euclidean space.
                Hopefully this interactive makes clear why it's not possible in
                4D, and why 6D is redundant (note the left axes traces a
                2-dimensional surface, thus reducing the required dimension by
                1))
            ''', '''
            ''', '''\
                This interactive always plots at least a single point on the
                Gr(1, 3) manifold. This point is controlled via spherical
                coordinates (under the projective equivalence) with the two
                sliders. A visualization of the selected projective space in 3D
                is shown in the top right corner as a line. Due to the
                projective equivalence nature of Gr(1, 3), all points may be
                represented using only theta <= 0.5 pi. However, the slider
                allows for theta to range from 0 to pi to allow for more
                diverse visualizations of trajectories. To highlight this
                projective behavior, the scene in the top right corner shades
                the positive hemisphere darker than the negative hemisphere.
            ''', '''
            ''', '''\
                In addition, there are two buttons which toggle certain visuals from being displayed: 
            ''', '''\
                * The boundary of the mobius strip defined by (pi/2 - x) <=
                theta <= (pi/2 + x). When this boundary loop is visible
                (purple), so too will be the fiber associated to phi on this
                strip (blue).
            ''', '''\
                * The entire Gr(1, 3) manifold, or only the mobius strip
                mentioned above.
            ''', '''
            ''', '''\
                Users are encouraged to play amply with the sliders to
                visualize various trajectories, deform mobius
                strips/boundaries, and consider how the product of the two 3D
                plots evades self-intersection of the Gr(1, 3) topology. Users
                are also encouraged to consider the charts/projections one
                could make to 'naturally' map this geometry, given its
                beautiful structure.
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

