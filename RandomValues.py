from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tabels_form import *


class Sample:
    def __init__(self, values):
        self.values = values
        self.sample_mean = np.mean(self.values)
        self.sample_dispersion = np.mean([(i - self.sample_mean) ** 2 for i in self.values])


class RandomValues(Sample):
    def __init__(self, values, vb=False):
        super().__init__(values)
        self.sample_choice = False
        self.sample = self.create_sample() if vb else False
        self.assorted_sample_dispersion = len(self.values) / (len(self.values) - 1) * self.sample_dispersion
        self.initial_moment = self.sample_moment(0, 4)
        self.mean_moment = self.sample_moment(self.sample_mean, 4)
        self.sample_standard_deviation = np.sqrt(self.sample_dispersion)
        self.variation_coefficient = self.sample_standard_deviation / self.sample_mean
        self.selective_asymmetry_coefficient = (self.mean_moment[2] - 3 * self.mean_moment[0] * self.mean_moment[1] +
                                                self.mean_moment[0] ** 3 * 2) / (
                                                   np.sqrt(self.sample_standard_deviation ** 3))
        self.selective_excess = (self.mean_moment[3] - 4 * self.mean_moment[0] * self.mean_moment[2] + 6 *
                                 self.mean_moment[0] ** 2 * self.mean_moment[1] - 3 * self.mean_moment[0] ** 4) / (
                                        self.sample_dispersion ** 2) - 3
        self.dependence = None
        self.confidence_intervals_math_e, self.confidence_intervals_dispersion = self.d_confidence_intervals()
        self.distribution_type = self.find_distribution_type()

    def create_sample(self, sample_choice=False):
        if not sample_choice:
            self.sample_choice = np.random.randint(0, 100, len(self.values))
            for i in range(len(self.sample_choice)):
                if self.sample_choice[i] < 50:
                    self.sample_choice[i] = 1
                else:
                    self.sample_choice[i] = 0
        else:
            self.sample_choice = sample_choice
        temp = []
        for i in range(100):
            if self.sample_choice[i]:
                temp.append(self.values[i])
        if abs(len(temp) - len(self.values) / 2) < len(self.values) / 20:
            sample = RandomValues(temp)
        else:
            sample = self.create_sample()

        return sample

    def sample_moment(self, C, K):
        moment = []
        for i in range(1, K + 1):
            moment.append(np.mean([(j - C) ** i for j in self.values]))
        return moment

    def correlation(self, second):
        temp = np.mean([self.values[i] * second.values[i] - self.sample_mean * second.sample_mean for i in
                        range(len(self.values))]) / np.sqrt(
            self.sample_dispersion * second.sample_dispersion)
        tt = (temp * np.sqrt(len(self.values) - 2)) / np.sqrt(1 - temp ** 2)
        self.dependence = f'Linear dependency, t = {tt}' if tt > 1.99 else None

        return temp

    def to_exel(self, way='result.xlsx'):
        temp = len(self.values) // 4
        result = []
        for i in range(4):
            result.append(self.values[0 + temp * i: 0 + temp * i + temp])
            if type(self.sample_choice) != bool:
                result.append(self.sample_choice[0 + temp * i: 0 + temp * i + temp])
        pd.DataFrame(result).transpose().to_excel(way)

    def histogram(self, columns=7):
        values = copy(self.values)
        values.sort()
        step = (values[-1] - values[0]) / columns
        j = 0
        ret = {}
        t = []
        ax = plt.subplots()[1]
        x_marks = [values[0] + step * i / 2 for i in range(columns * 2 + 1)]
        ax.set_xticks(x_marks)
        vl = len(values)
        intervals = {}
        for i in range(columns):
            temp = 0
            key = f'{values[0] + step * i}<=x<{values[0] + step + step * i}' if i + 1 != columns else \
                f'{values[0] + step * i}<=x<={values[0] + step + step * i}'
            for x in values:
                if eval(key):
                    temp += 1

            intervals.update({key: temp})
        iter_number = 0
        for i in intervals:
            ax.bar(values[0] + step / 2 + step * iter_number, intervals[i], color='white', width=step, edgecolor='b')
            ret.update({
                f'{round(values[0] + step * iter_number, 2)}--{round(values[0] + step + step * iter_number, 2)}': f'{intervals[i]}/{vl}'})
            iter_number += 1

        # for i in range(columns):
        #     temp = 0
        #     while values[j] < values[0] + step + step * i:
        #         temp += 1
        #         j += 1
        #
        #     ax.bar(values[0] + step / 2 + step * i,
        #            temp / vl if i != 6 else (temp + 1) / vl, color='white', width=step, edgecolor='b')
        #     ret.update({
        #         f'{round(values[0] + step * i, 2)}-{round(values[0] + step + step * i, 2)}': \
        #             f'{temp}/{vl}' if i != 6 else f'{temp + 1}/{vl}'})
        #     t.append(temp)

        #
        n = []
        for i in t:
            if i not in n:
                n.append(i)

        ax.set_yticks([i / vl for i in n])
        x_labels = [round(x_marks[i], 2) if i % 2 == 0 else '-' for i in range(columns * 2 + 1)]
        ax.set_xticklabels(x_labels)
        #ax.set_yticklabels([ret[i] for i in ret])

        return plt, ret

    def d_confidence_intervals(self):
        mat_inter = (self.sample_mean - 1.98 * self.sample_standard_deviation / np.sqrt(len(self.values)),
                     self.sample_mean + 1.98 * self.sample_standard_deviation / np.sqrt(len(self.values)))
        dis_inter = (100 * self.sample_dispersion / 129.5612,
                     100 * self.sample_dispersion / 74.22193)  # TODO переделать универсальную если будет неоюходимоость
        return mat_inter, dis_inter

    def stat_func(self):
        values = copy(self.values)
        values.sort()
        step = (values[-1] - values[0]) / 7
        j = 0
        ret = [[], [[], []]]
        ax = plt.subplots()[1]
        for i in range(6):
            temp = 0
            while values[j] < values[0] + step + step * i:
                temp = temp + 1
                j += 1
            ax.hlines(xmin=round(values[0] + step * i, 2), xmax=round(values[0] + step * i + step, 2),
                      y=j / len(values), color='b',
                      linewidth=3)
            ax.scatter(round(values[0] + step * i, 2), j / len(values), color='#a3c4dc')
            ax.scatter(round(values[0] + step * i + step, 2), j / len(values), color='#0e668b')
            ret[0].append(round(values[0] + step * i, 2))
            ret[1][0].append(f'{j}/{len(values)}')
            ret[1][1].append(j / len(values))
        ax.hlines(xmin=round(values[0] + step * 6, 2), xmax=round(values[0] + step * 6 + step + 0.1, 2), y=1, color='b',
                  linewidth=3)
        ax.scatter(round(values[0] + step * 6, 2), 1, color='#a3c4dc')
        ret[0].append(round(values[0] + step * 6, 2))
        ret[1][0].append(f'{len(values)}/{len(values)}')
        ret[1][1].append(1)

        ax.set_yticks(ret[1][1])
        ax.set_yticklabels(ret[1][0])
        ax.set_xticks(ret[0])
        ax.set_xticklabels(ret[0])

        return plt, ret

    def func(self, other):
        """
        self.values -  все значения x
        self.sample_mean - выборочное следнее х
        self.sample_dispersion - выборочная дисперсия
        other.* аналогичо для y

        :param other:
        :return:
        """
        result = []
        count = []
        plt.cla()
        for i in range(len(self.values)):
            temp = [self.values[i], other.values[i]]
            if temp in result:
                tt = result.index(temp)
                count[tt] = count[tt] + 20
            else:
                result.append(temp)
                count.append(10)

        for i in range(len(result)):
            plt.scatter(result[i][0], result[i][1], color='blue', s=count[i])
        a_dot = (np.mean(
            [self.values[i] * other.values[i] for i in
             range(len(self.values))]) - self.sample_mean * other.sample_mean) / self.sample_dispersion
        b_dot = other.sample_mean - a_dot * self.sample_mean
        # y = ax+b
        plt.plot([min(self.values), max(self.values)],
                 (a_dot * min(self.values) + b_dot, a_dot * max(self.values) + b_dot), c='r')
        plt.minorticks_on()

        #  Определяем внешний вид линий основной сетки:
        plt.grid(which='major',
                 color='k',
                 linewidth=1)

        #  Определяем внешний вид линий вспомогательной
        #  сетки:
        plt.grid(which='minor',
                 color='k',
                 linestyle=':')

        plt.show()
        return a_dot, b_dot

    def find_distribution_type(self, _h=5):
        keys = Laplas_function.keys()
        temp = self.histogram(_h)[1]
        inter_values = [eval(temp[i]) for i in temp]
        temp = [i.split('--') for i in temp]
        tt = []
        for i in temp:
            if i[0] not in tt:
                tt.append(i[0])
        tt.append(temp[-1][-1])
        intervals = [float(i) for i in tt]
        r = []  # Мат, величина r
        for i in range(1, len(intervals)):
            lap1 = round((intervals[i] - self.sample_mean) / np.sqrt(self.sample_dispersion), 2)
            if abs(lap1) not in keys:
                lap1 = round(lap1 - 0.01, 2)
            lap1 = Laplas_function[lap1] if lap1 > 0 else -Laplas_function[-lap1]
            lap2 = round((intervals[i - 1] - self.sample_mean) / np.sqrt(self.sample_dispersion), 2)
            if abs(lap2) not in keys:
                lap2 = round(lap2 - 0.01, 2)
            lap2 = Laplas_function[lap2] if lap2 > 0 else -Laplas_function[-lap2]
            r.append(lap1 - lap2)

        return 100 * sum([(inter_values[i] - r[i]) ** 2 for i in range(len(inter_values))])
