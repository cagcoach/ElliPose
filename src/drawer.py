import matplotlib.pyplot as plt


class drawer():
    def __init__(self):
        pass

    @staticmethod
    def drawTwoSkeletons(posesA,posesB,keypoints_metadata, azim=0, elev=15):

        plt.ioff()
        fig = plt.figure(figsize=(6 * (1 + len(posesA)), 6))
        ax_in = fig.add_subplot(1, 1 + len(posesA), 1)
        ax_in.get_xaxis().set_visible(False)
        ax_in.get_yaxis().set_visible(False)
        ax_in.set_axis_off()
        ax_in.set_title('Input')

        ax_3d = []
        lines_3d = []
        trajectories = []
        radius = 1.7
        for index, (title, data) in enumerate(posesA.items()):
            ax = fig.add_subplot(1, 1 + len(posesA), index + 2, projection='3d')
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim3d([-radius / 2, radius / 2])
            ax.set_zlim3d([0, radius])
            ax.set_ylim3d([-radius / 2, radius / 2])
            try:
                ax.set_aspect('equal')
            except NotImplementedError:
                ax.set_aspect('auto')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.dist = 7.5
            ax.set_title(title)  # , pad=35
            ax_3d.append(ax)
            lines_3d.append([])
            trajectories.append(data[:, 0, [0, 1]])
        poses = list(posesA.values())

        initialized = False
        image = None
        lines = []
        points = None


        parents = skeleton.parents()

        def update_video(i):
            nonlocal initialized, image, lines, points

            for n, ax in enumerate(ax_3d):
                ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
                ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])

            print('{}/{}      '.format(i, limit), end='\r')

        fig.tight_layout()

        anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
        if output.endswith('.mp4'):
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
            anim.save(output, writer=writer)
        elif output.endswith('.gif'):
            anim.save(output, dpi=80, writer='imagemagick')
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
        plt.close()