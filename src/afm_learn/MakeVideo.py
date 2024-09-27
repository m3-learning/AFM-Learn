import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from matplotlib import gridspec
from cmap import ScientificColourMaps
from scipy.signal import find_peaks
from Viz import show_images

def generate_z_voltages(voltage_start, voltage_amplitude, voltage_step, frames_per_voltage=1, repeat_times=1, voltage_end=0, viz=False):
    voltages = np.concatenate([np.arange(voltage_start, voltage_amplitude, voltage_step), np.arange(voltage_amplitude, -voltage_amplitude, -voltage_step), np.arange(-voltage_amplitude, voltage_end+voltage_step, voltage_step)])
    if frames_per_voltage > 1:
        voltages = [0] + list(np.repeat(voltages[1:], frames_per_voltage))
        voltages = np.array(voltages)
    if repeat_times > 1:
        if voltages[0] == 0:
            voltages = [0] + list(np.tile(voltages[1:], repeat_times))
            voltages = np.array(voltages)
        else:
            voltages = np.tile(voltages, repeat_times)
    
    voltage_labels = [f'Frame:{i}, Voltage:{voltages[i]}V' for i in range(len(voltages))]
    if viz:
        plt.figure(figsize=(10, 2))
        plt.plot(voltages)
        plt.xlabel('Frame')
        plt.ylabel('Voltage (V)')
        plt.grid()
        plt.show()
    return voltages, voltage_labels



class VideoMaker:
    def __init__(self, fps, x_label='Frame', colormap_dict='roma', dynamic_clim=None, custom_clim=None):

        self.fps = fps
        self.x_label = x_label
        self.colormap_dict = colormap_dict
        self.dynamic_clim = dynamic_clim if dynamic_clim is not None else []
        self.custom_clim = custom_clim

    def make_video(self, frame_sequences_dict, voltages, video_name=None, voltage_labels=None, x=None):
        self.frame_seq_length = len(frame_sequences_dict)
        self.frame_sequences = list(frame_sequences_dict.values())
        self.seq_names = list(frame_sequences_dict.keys())

        self.voltages = voltages
        self.labels = voltage_labels
        if isinstance(self.labels, type(None)):
            self.labels = [f'{self.x_label}: {i}, Voltage: {v:.2f} V' for i, v in enumerate(voltages)]

        self.video_name = video_name
        if self.video_name is None:
            self.video_name = 'output_video.mp4'
            
        if all(e is None for e in [None, None, None, None, (-90, 280), None]):
            self.custom_clim = [None] * self.frame_seq_length
        else:
            self.custom_clim = self.custom_clim
        print(self.custom_clim)

        self.x = np.arange(0, len(voltages)) if x is None else x

        self.extend_frames()
        self.setup_plots()
        self.animate_video()

    def extend_frames(self):
        n_extend_frames = 5
        for i in range(self.frame_seq_length):
            extra_frames = np.repeat(self.frame_sequences[i][-1][np.newaxis, :, :], n_extend_frames, axis=0)
            self.frame_sequences[i] = np.concatenate((self.frame_sequences[i], extra_frames), axis=0)
        self.voltages = np.concatenate((self.voltages, np.repeat(self.voltages[-1], n_extend_frames)))
        self.x = np.concatenate((self.x, np.repeat(self.x[-1], n_extend_frames)))
        self.labels = np.concatenate((self.labels, np.repeat(self.labels[-1], n_extend_frames)))

        self.voltage_colors = self.load_cmap('Default')(np.linspace(0, 1, len(self.voltages)))
        self.voltage_colors_front = np.array([self.voltage_colors[0]]*n_extend_frames)
        self.voltage_colors = np.concatenate((self.voltage_colors_front, self.voltage_colors))

    def load_cmap(self, name):
        if name == 'Phase':
            cmap = self.colormap_dict['Phase']
        else:
            cmap = self.colormap_dict['Default']
        return cmap
    
    def setup_plots(self):
        cols = 3 if self.frame_seq_length <= 3 or self.frame_seq_length == 6 else 4
        # cols = min(max(3, self.frame_seq_length), 4)
        rows = (self.frame_seq_length + cols - 1) // cols  # Ceiling division for image rows
        # print(cols, rows)
        figsize = (15, 1 + 4 * rows * 2)  # Double the rows to include histograms
        fig = plt.figure(figsize=figsize)
        self.fig = fig
        gs = gridspec.GridSpec(rows * 2 + 1, cols, height_ratios=[0.6] + [1, 0.5] * rows)

        ax1 = fig.add_subplot(gs[0, :])  # Voltage plot
        ax1.plot(self.x, self.voltages, c='k', zorder=0)
        self.im_scatter = ax1.scatter(self.x[0], self.voltages[0], color=self.voltage_colors[0], zorder=1)
        ax1.set_ylabel('Voltage')
        ax1.set_xlabel(self.x_label)
        ax1.tick_params(axis='x', direction='in')
        ax1.tick_params(axis='y', direction='in')

        self.im_list = []
        self.hist_list = []
        self.cb_list = []
        for i, (frames, name, clim) in enumerate(zip(self.frame_sequences, self.seq_names, self.custom_clim)):
            # print(i)
            ax_img = fig.add_subplot(gs[1 + 2 * (i // cols), i % cols])
            cmap = self.load_cmap(name)
            im = ax_img.imshow(frames[0], cmap=cmap)
            # im.set_clim(*(clim if clim else (np.min(frames), np.max(frames))))

            cb = self.fig.colorbar(im, ax=ax_img)
            # fig.colorbar(im, ax=ax_img)
            ax_img.set_title(self.seq_names[i])
            ax_img.axis('off')

            ax_hist = fig.add_subplot(gs[2 + 2 * (i // cols), i % cols])
            hist, bins = np.histogram(frames[0].flatten(), bins=50)
            hist_line, = ax_hist.plot(bins[:-1], hist, linestyle='-', color='blue')
            ax_hist.set_ylabel('Count')
            ax_hist.set_xlabel('Pixel value')

            # not use dynamic clim for label in dynamic_clim
            if name not in self.dynamic_clim:
                if clim:
                    im.set_clim(*clim)
                    ax_hist.set_xlim(clim)
                else:
                    # mean, std = np.mean(frames), np.std(frames)
                    # range = (mean - 3 * std, mean + 3 * std)
                    range = (np.min(frames), np.max(frames))
                    im.set_clim(*range)
                    ax_hist.set_xlim(np.min(frames), np.max(frames))
            else:
                im.set_clim(np.min(frames[0]), np.max(frames[0]))
                ax_hist.set_xlim(np.min(frames[0]), np.max(frames[0]))


            # hist_line.axes.set_xlim(bins[0], bins[-1])  # Update x-axis range to match histogram bins
            # hist_line.axes.set_ylim(hist.min(), hist.max())  # Update y-axis range to cover new histogram heights
            # ax_hist.set_xlim((-50, 230))
            # print(clim)

            self.im_list.append(im)
            self.hist_list.append(hist_line)
            self.cb_list.append(cb)
        self.im_list = [self.im_scatter] + self.im_list

    def animate_video(self):
        ani = FuncAnimation(self.fig, self.update, frames=len(self.frame_sequences[0]),
                            fargs=[self.frame_sequences, self.im_list, self.cb_list, self.hist_list, self.x, self.voltages, self.voltage_colors], blit=True)
        writer = FFMpegWriter(fps=self.fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
        ani.save(self.video_name, writer=writer)
        plt.tight_layout(pad=1)
        plt.close(self.fig)
        # plt.show()

    def update(self, frame_number, frame_sequences, im_list, cb_list, hist_list, x, voltages, colors):
        wrapped_frame_number = frame_number % len(voltages)
        im_list[0].set_offsets([x[wrapped_frame_number], voltages[wrapped_frame_number]])
        im_list[0].set_facecolor(colors[wrapped_frame_number])
        for im, cb, hist_line, frames, name in zip(im_list[1:], cb_list, hist_list, frame_sequences, self.seq_names):
            frame = frames[wrapped_frame_number]
            im.set_data(frame)
            hist, bins = np.histogram(frame.flatten(), bins=50)
            hist_line.set_xdata(bins[:-1])  # Update histogram bins
            hist_line.set_ydata(hist)  # Update histogram

            if name in self.dynamic_clim:
                # Update color limits
                im.autoscale()
                cb.update_normal(im)

                # Update histogram
                # Set the x-axis limits to the range of bins
                hist_line.axes.set_xlim([bins.min(), bins.max()])
                
            # always update histogram's y axis limits to the range of histogram counts
            hist_line.axes.set_ylim([0, hist.max() + np.sqrt(hist.max())])  # Adding some padding

        title = os.path.basename(self.video_name).split('.')[0] + '\n' + self.labels[wrapped_frame_number]
        plt.suptitle(title, fontsize=12)
        return im_list + hist_list

# Example usage:
# video_maker = VideoMaker(frame_sequences, seq_names, voltages, 'output_video.mp4', 30)
# video_maker.make_video()
