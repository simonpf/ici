from netCDF4 import Dataset
from scipy   import stats
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c


class IciData:
    def __load_data__(self, filename, version):
        data = Dataset(filename, "r", format="NETCDF4")

        self.name    = "orignal"
        self.version = version

        # Simulation Properties

        self.iwp                 = data.variables['iwp'][:]
        self.dmean               = data.variables['dmean'][:]
        self.zcloud              = data.variables['zcloud'][:]
        self.lwp                 = data.variables['lwp'][:]
        self.rwp                 = data.variables['rwp'][:]
        self.cwv                 = data.variables['cwv'][:]
        self.surface_pressure    = data.variables['surface_pressure'][:]
        self.surface_temperature = data.variables['surface_temperature'][:]
        self.surface_wind_speed  = data.variables['surface_wind_speed'][:]
        self.weight              = data.variables['weight'][:]

        self.icehabit            = data.variables['icehabit'][:]
        self.surface_type        = data.variables['surface_type'][:]

        # Channels
        if (version < 3):
            var_name_tb = "tb_fs_ch_"
            var_name_od = "od_cs_ch_"
        else:
            var_name_tb = "dtb_ch_"
            var_name_od = "od_ch_"

        self.ch_1  = data.variables[var_name_tb + str(1)][:]
        self.ch_2  = data.variables[var_name_tb + str(2)][:]
        self.ch_3  = data.variables[var_name_tb + str(3)][:]
        self.ch_4  = data.variables[var_name_tb + str(4)][:]
        self.ch_5  = data.variables[var_name_tb + str(5)][:]
        self.ch_6  = data.variables[var_name_tb + str(6)][:]
        self.ch_7  = data.variables[var_name_tb + str(7)][:]
        self.ch_8  = data.variables[var_name_tb + str(8)][:]
        self.ch_9  = data.variables[var_name_tb + str(9)][:]
        self.ch_10 = data.variables[var_name_tb + str(10)][:]
        self.ch_11 = data.variables[var_name_tb + str(11)][:]

        self.od_ch_1   = data.variables[var_name_od + str(1)][:]
        self.od_ch_2   = data.variables[var_name_od + str(2)][:]
        self.od_ch_3   = data.variables[var_name_od + str(3)][:]
        self.od_ch_4   = data.variables[var_name_od + str(4)][:]
        self.od_ch_5   = data.variables[var_name_od + str(5)][:]
        self.od_ch_6   = data.variables[var_name_od + str(6)][:]
        self.od_ch_7   = data.variables[var_name_od + str(7)][:]
        self.od_ch_8   = data.variables[var_name_od + str(8)][:]
        self.od_ch_9   = data.variables[var_name_od + str(9)][:]
        self.od_ch_10  = data.variables[var_name_od + str(10)][:]
        self.od_ch_11  = data.variables[var_name_od + str(11)][:]

        # Deal with inf values...
        od_max = 10000.0
        self.od_ch_1[np.isinf(self.od_ch_1)] = od_max
        self.od_ch_2[np.isinf(self.od_ch_2)] = od_max
        self.od_ch_3[np.isinf(self.od_ch_3)] = od_max
        self.od_ch_4[np.isinf(self.od_ch_4)] = od_max
        self.od_ch_5[np.isinf(self.od_ch_5)] = od_max
        self.od_ch_6[np.isinf(self.od_ch_6)] = od_max
        self.od_ch_7[np.isinf(self.od_ch_7)] = od_max
        self.od_ch_8[np.isinf(self.od_ch_8)] = od_max
        self.od_ch_9[np.isinf(self.od_ch_9)] = od_max
        self.od_ch_10[np.isinf(self.od_ch_10)] = od_max
        self.od_ch_11[np.isinf(self.od_ch_11)] = od_max

        self.channels = [self.ch_1, self.ch_2, self.ch_3, self.ch_4, self.ch_5, self.ch_6,
                         self.ch_7, self.ch_8, self.ch_9, self.ch_10, self.ch_11]
        self.channels_od  = [self.od_ch_1,  self.od_ch_2,  self.od_ch_3,  self.od_ch_4,  self.od_ch_5,  self.od_ch_6,
                             self.od_ch_7,  self.od_ch_8,  self.od_ch_9,  self.od_ch_10,  self.od_ch_11]

        self.t_max = max(map(np.max, self.channels))
        self.t_min = min(map(np.min, self.channels))
        self.od_max  = max(map(np.max, self.channels_od))
        self.od_min  = min(map(np.min, self.channels_od))

        data.close()

        # Additional properties

        self.noise = np.asarray([0.8, 0.8, 0.8, 0.7, 1.2, 1.3, 1.5, 1.4, 1.6, 2.0, 1.6])
        self.fs    = np.asarray([183.31, 183.31, 183.31, 243.2, 325.15, 325.15, 325.15, 448.0, 448.0, 448.0, 664.2])
        self.n     = self.iwp.shape[0]

    def __init__(self, filename = None):
        if not(filename):
            filename = "/home/simonpf/projects/ici/data/ici-db-v2.nc"
        self.__load_data__(filename, 2)

    def __getitem__(self, indices):
        new_data = IciData.__new__(IciData)

        new_data.version = self.version

        new_data.iwp                 = self.iwp[indices]
        new_data.dmean               = self.dmean[indices]
        new_data.zcloud              = self.zcloud[indices]
        new_data.lwp                 = self.lwp[indices]
        new_data.rwp                 = self.rwp[indices]
        new_data.cwv                 = self.cwv[indices]
        new_data.surface_pressure    = self.surface_pressure[indices]
        new_data.surface_temperature = self.surface_temperature[indices]
        new_data.surface_wind_speed  = self.surface_wind_speed[indices]
        new_data.weight              = self.weight[indices]

        new_data.icehabit            = self.icehabit[indices]
        new_data.surface_type        = self.surface_type[indices]

        # Channels

        new_data.ch_1  = self.ch_1[indices]
        new_data.ch_2  = self.ch_2[indices]
        new_data.ch_3  = self.ch_3[indices]
        new_data.ch_4  = self.ch_4[indices]
        new_data.ch_5  = self.ch_5[indices]
        new_data.ch_6  = self.ch_6[indices]
        new_data.ch_7  = self.ch_7[indices]
        new_data.ch_8  = self.ch_8[indices]
        new_data.ch_9  = self.ch_9[indices]
        new_data.ch_10 = self.ch_10[indices]
        new_data.ch_11 = self.ch_11[indices]

        new_data.od_ch_1  = self.od_ch_1[indices]
        new_data.od_ch_2  = self.od_ch_2[indices]
        new_data.od_ch_3  = self.od_ch_3[indices]
        new_data.od_ch_4  = self.od_ch_4[indices]
        new_data.od_ch_5  = self.od_ch_5[indices]
        new_data.od_ch_6  = self.od_ch_6[indices]
        new_data.od_ch_7  = self.od_ch_7[indices]
        new_data.od_ch_8  = self.od_ch_8[indices]
        new_data.od_ch_9  = self.od_ch_9[indices]
        new_data.od_ch_10 = self.od_ch_10[indices]
        new_data.od_ch_11 = self.od_ch_11[indices]

        new_data.channels = [new_data.ch_1, new_data.ch_2, new_data.ch_3, new_data.ch_4, new_data.ch_5, new_data.ch_6,
                             new_data.ch_7, new_data.ch_8, new_data.ch_9, new_data.ch_10, new_data.ch_11]
        new_data.channels_od  = [new_data.od_ch_1,  new_data.od_ch_2,  new_data.od_ch_3,  new_data.od_ch_4,  new_data.od_ch_5,  new_data.od_ch_6,
                                 new_data.od_ch_7,  new_data.od_ch_8,  new_data.od_ch_9,  new_data.od_ch_10,  new_data.od_ch_11]

        new_data.t_max = max(map(np.max, new_data.channels))
        new_data.t_min = min(map(np.min, new_data.channels))
        new_data.od_max  = max(map(np.max, new_data.channels_od))
        new_data.od_min  = min(map(np.min, new_data.channels_od))

        new_data.noise = self.noise
        new_data.fs    = self.fs
        new_data.n     = new_data.iwp.shape[0]

        return new_data

    def clear_sky(self):
       clear_sky = self[self.iwp == 0.0]
       return clear_sky

    def tropics(self):
       tropics = self[self.cwv > 40.0]
       return tropics

    def plot_overview(self, include_general = False):

        if (include_general):
            data = IciData()

        # IWP Histogram
        if (include_general):
            ax = plt.gca()
            n1, bins1, patches1 = ax.hist(data.iwp, 50, label="Complete Dataset", alpha = 0.5, zorder=1, color="C0")
            ax.hist(self.iwp, bins1, label="IWP", zorder=10, color="C0")
        else:
            n1, bins1, patches1 = plt.hist(self.iwp, 50, label="IWP", zorder=10)
        ax = plt.gca()
        ax.set_yscale("log")
        if (self.iwp.min() == 0.0):
            ax.hist(self.iwp[self.iwp == 0.0], bins1[:2], label="Clear Sky", color="C1", zorder=10)
        ax.set_xlim([0.0, bins1[-1]])
        plt.legend()
        plt.title("IWP Distribution")

        # CWV
        plt.figure()
        if (include_general):
            ax = plt.gca()
            n1, bins1, patches1 = ax.hist(data.cwv, 50, label="Complete Dataset", alpha = 0.5, zorder=1, color="C0")
            ax.hist(self.cwv, bins1, label="CWV", zorder=10, color="C0")
        else:
            n1, bins1, patches1 = plt.hist(self.cwv, 50, label="CWV", zorder=10)
        ax = plt.gca()
        ax.set_yscale("log")
        ax.set_xlim([0.0, bins1[-1]])
        plt.legend()
        plt.title("WVC Distribution")

        # ground Types
        plt.figure()
        plt.title("Surface Type Distribution")
        n, bins, patches = plt.hist(self.surface_type, 2, label="Surface Type", zorder=10)
        if (include_general):
            n_o, bins_o, patches_o = plt.hist(data.surface_type, 2, alpha = 0.5, zorder=1, color="C0")
            patches_o[-1].set_color("C1")

        ax = plt.gca()
        plt.xticks([0.25, 0.75], ["Sea (0)", "Ground (1)"])
        ax.set_xlim([0.0,1.0])
        patches[-1].set_color("C1")

        # dtb
        f, axs = plt.subplots(4,3)
        for i in range(0,4):
            for j in range(0,3):
                ci = i * 3 + j
                if (ci < 11):
                    ax = axs[i,j]
                    n, bins, patches = ax.hist(self.channels[ci], 100, range = [self.t_min, self.t_max], zorder=10)
                    if (include_general):
                        ax.hist(data.channels[ci], bins=bins, alpha = 0.5, zorder=1, color="C0")
                    ax.set_yscale("log")
                    # This is too slow...
                    #kernel = stats.gaussian_kde(self.channels[ci])
                    #ax.plot(n, kernel(n), c="C1")
                    ax.set_xlabel("$T_B [K]$")
                    ax.set_ylabel("n")
                    ax.set_title("Channel " + str(ci + 1))
        axs[3,2].axis("off")
        plt.tight_layout()

        # od
        f, axs = plt.subplots(4,3)
        for i in range(0,4):
            for j in range(0,3):
                ci = i * 3 + j
                if (ci < 11):
                    ax = axs[i,j]
                    n, bins, patches = ax.hist(self.channels_od[ci], 100, range = [self.od_min, self.od_max])
                    if (include_general):
                        ax.hist(data.channels_od[ci], bins=bins, alpha = 0.5, zorder=1, color="C0")
                    ax.set_yscale("log")
                    # This is too slow...
                    #kernel = stats.gaussian_kde(self.channels[ci])
                    #ax.plot(n, kernel(n), c="C1")
                    ax.set_xlabel(r"$\tau$")
                    ax.set_ylabel("n")
                    ax.set_title("Channel " + str(ci + 1))
        axs[3,2].axis("off")
        plt.tight_layout()

        # od
        # bins_x = np.linspace(self.od_min, self.od_max, 101)
        # bins_y = np.linspace(self.iwp.min(), self.iwp.max(), 101)
        # f, axs = plt.subplots(4,3)
        # cs_max = 0
        # for i in range(0,4):
        #     for j in range(0,3):
        #         ci = i * 3 + j
        #         if (ci < 11):
        #             ax = axs[i,j]
        #             cs, xedges, yedges, img = axs[i,j].hist2d(self.channels_od[ci], self.iwp, [bins_x, bins_y])
        #             cs_max = max(cs_max, cs.max())
        # norm = c.LogNorm(1, cs_max, clip=True)
        # for i in range(0, 4):
        #     for j in range (0,3):
        #         ci = i * 3 + j
        #         if (ci < 11):
        #             ax = axs[i,j]
        #             ax.hist2d(self.channels_od[ci], self.iwp, [bins_x, bins_y], norm=norm)
        #             ax.set_xlabel(r"$\tau$")
        #             ax.set_ylabel("IWP")
        #             ax.set_title("Channel " + str(ci + 1))
        # axs[3,2].axis("off")
        plt.tight_layout()

    def tb_mat(self):
        tb_mat = np.stack(self.channels)
        return tb_mat

    def cov_mat(self):
        tb_mat  = self.tb_mat()
        tb_mean = tb_mat.mean(axis=1)
        tb_mat  = np.transpose(np.transpose(tb_mat) - np.transpose(tb_mean))
        s = np.dot(tb_mat, np.transpose(tb_mat))

        sigma    = np.sqrt(s.diagonal())
        s_normed = s / np.outer(sigma, sigma)
        return s, s_normed

    def svd(self, prefix = ""):
        tb_mat = np.stack(self.channels)
        tb_mean = tb_mat.mean(axis=1)
        tb_mat  = np.transpose(np.transpose(tb_mat) - np.transpose(tb_mean)) / np.sqrt(self.n - 1)
        u, s, v = np.linalg.svd(tb_mat, full_matrices = False)
        np.save("data/" + prefix + "u.npy", u)
        np.save("data/" + prefix + "s.npy", s)
        return s, np.transpose(u), np.sqrt(np.dot(np.transpose(u ** 2), self.noise ** 2))

    def plot_svd(self, prefix = ""):
        try:
            u = np.load("data/" + prefix + "u.npy")
            s = np.load("data/" + prefix + "s.npy")
        except e:
            self.svd(prefix)
            print("Could not load SVD.")
            u = np.load("data/" + prefix + "u.npy")
            s = np.load("data/" + prefix + "s.npy")
        ns = np.sqrt(np.dot(np.transpose(u ** 2), self.noise ** 2))
        plt.figure()
        plt.plot(s , label =  "Singular Values")
        plt.plot(ns, label = r"$NE \Delta T$")
        ax = plt.gca()
        ax.set_xlabel("PCA Component")
        ax.set_ylabel(r"$\sigma [K]$")
        plt.legend()

        x = np.arange(1,12)
        f, axs = plt.subplots(4,3)
        for i in range(0,4):
            for j in range(0,3):
                ci = i * 3 + j
                if (ci < 11):
                    ax = axs[i,j]
                    ax.plot(x, u[:, ci], ls='-', marker='o')
                    ax.set_xlabel("Channel Frequency [GHz]")
                    ax.set_ylabel("Channel Weight")
                    ax.set_title("Channel " + str(ci))
                    ax.xaxis.set_ticks(np.arange(1,12))
                    ax.set_xlim([0.5, 11.5])
                    ax.set_xticklabels([str(np.round(self.fs[i])) for i in np.arange(0,11)], rotation=45)
        axs[3,2].axis("off")
        plt.tight_layout()

    def save(self, filename):
        p = Path(filename)
        data = Dataset(p, "w", format="NETCDF4")
        time = data.createDimension("time", self.n)
        # Integer data.
        print(self.n)
        print(self.icehabit.shape)
        icehabit                 = data.createVariable("icehabit", "i4", ("time",))
        surface_type             = data.createVariable("surface_type", "i4", ("time",))
        icehabit[:]     = self.icehabit
        surface_type[:] = self.surface_type

        # Floating Point Data
        iwp                 = data.createVariable("iwp", "f8", ("time",))
        dmean               = data.createVariable("dmean", "f8", ("time",))
        zcloud              = data.createVariable("zcloud", "f8", ("time",))
        lwp                 = data.createVariable("lwp", "f8", ("time",))
        rwp                 = data.createVariable("rwp", "f8", ("time",))
        cwv                 = data.createVariable("cwv", "f8", ("time",))
        surface_pressure    = data.createVariable("surface_pressure", "f8", ("time",))
        surface_temperature = data.createVariable("surface_temperature", "f8", ("time",))
        surface_wind_speed  = data.createVariable("surface_wind_speed", "f8", ("time",))
        weight              = data.createVariable("weight", "f8", ("time",))

        iwp[:]                 = self.iwp
        dmean[:]               = self.dmean
        zcloud[:]              = self.zcloud
        lwp[:]                 = self.lwp
        rwp[:]                 = self.rwp
        cwv[:]                 = self.cwv
        surface_pressure[:]    = self.surface_pressure
        surface_temperature[:] = self.surface_temperature
        surface_wind_speed[:]  = self.surface_wind_speed
        weight[:]              = self.weight

        # Channels
        if (self.version < 3):
            var_name_tb = "tb_fs_ch_"
            var_name_od = "od_cs_ch_"
        else:
            var_name_tb = "dtb_ch_"
            var_name_od = "od_ch_"

        cs = [None] * 11
        for i in range(11):
            channel = data.createVariable(var_name_tb+str(i + 1), "f8", ("time",))
            channel[:] = self.channels[i]
            channel = data.createVariable(var_name_od+str(i + 1), "f8", ("time",))
            channel[:] = self.channels_od[i]

        data.close()

    def add_noise(self):
        for i in range(len(self.channels)):
            self.channels[i] = self.channels[i] + np.random.normal(loc=0.0, scale=self.noise[i], size=self.channels[i].shape)

    def split_data(self, name):
        inds   = np.random.permutation(self.n)
        ind_9  = int(np.round(self.n * 0.9))
        train_data = self[inds[:ind_9]]
        test_data  = self[inds[ind_9:]]
        p = Path("/home/simonpf/projects/ici/data/sets/" + name + "/train.nc")
        train_data.save(p)
        p = Path("/home/simonpf/projects/ici/data/sets/" + name + "/test.nc")
        test_data.save(p)

    def get_input_data(self, normalize=True):
        x = np.transpose(np.stack(self.channels))
        if (normalize):
            xmean = x.mean(axis=0)
            xvar  = x.var(axis=0)
            x = (x - xmean) / np.sqrt(xvar)
        return x

    def get_output_data(self, type):
        if (type == "clear_sky"):
            y = np.asarray((self.iwp == 0.0), dtype=np.int)
        return y






