# -*- coding: utf-8 -*-
import pandas as pd


class SensorValueAndTimeWithBayes():
    u"""時間とセンサの関係をベイズ推定で求めるための補助をするクラス

    p(z|t)だのp(t|z)だの，どんなふうに求まるのかをとにかく経験でわかるようにするため
    数式見てもわかったきになるだけだった
    """
    def __init__(self, filename):
        data = pd.read_csv(
            filename, delimiter=' ', header=None,
            names=('date', 'time', 'ir', 'lidar'))
        data['hour'] = [e // 10000 for e in data.time]

        data_grouped_by_hour = data.groupby('hour')
        u"""時間毎の頻度リスト

            時間がkey
            これを指定することがセンサ値zを時間tで条件付けることと同義
            まだ確率ではない，頻度
        """
        frequencies_by_hour = \
            {
                i : data_grouped_by_hour.lidar.get_group(i).value_counts().sort_index()
                for i in range(24)
            }
        u"""確率への変換

        こんなテーブルを作る
        同時確率分布P(z, t)

        z \ t           0              1              2     ...             23
        607    P(z=607,t=0)   P(z=607,t=1)   P(z=607,t=2)    ...   P(z=607,t=23)
        608    P(z=608,t=0)   P(z=608,t=1)   P(z=608,t=2)    ...   P(z=608,t=23)
        609    P(z=609,t=0)   P(z=609,t=1)   P(z=609,t=2)    ...   P(z=609,t=23)
        .
        .
        .
        """
        freq_dist = pd.concat(frequencies_by_hour, axis=1)
        freq_dist = freq_dist.fillna(0.)
        self._joint_z_t = freq_dist / len(data)

    def marginalize_in(self, base='z'):
        u"""周辺化

        一般に
            P(x) = sum_{y in Y} P(x, y)
        """
        if base != 'z' and base != 't':
            raise Exception(
                'z or t are only allowed specification for marginalization')

        if base == 'z':
            # zについて周辺化
            # 行毎の合計を出す（∵行がセンサ値を指定する要素だから）
            p_z = pd.DataFrame(self._joint_z_t.transpose().sum())
            return p_z[0]
        else:
            # tについて周辺化
            # 列毎の合計を出す（∵列が時間を指定する要素だから）
            # pandas.DataFrame.sumは列毎の総和をつくる
            p_t = pd.DataFrame(self._joint_z_t.sum())
            return p_t[0]

    def joint_z_t(self):
        return self._joint_z_t

    def cond_z_t(self):
        return self._joint_z_t / self.marginalize_in('t')

    def cond_t_z(self):
        return self._joint_z_t.transpose() / self.marginalize_in('z')


class BayesianFilter():
    u"""ベイズフィルタ

    p(y|x) = p(x|y)p(y) / p(x)
    """
    def __init__(self, cond_x_y):
        u"""学習した，原因を固定したときの観測値の条件付き確率を登録する

        Args:
            cond_x_y(pandas.DataFrame): 条件付き確率
        """
        self._cond_x_y = cond_x_y

    def update(self, x, p_y):
        u"""観測値と現在の原因分布から，より確かな原因分布へ更新する

        Args:
            x(float): 観測値
            p_y(pd.DataFrame): 原因の分布
        """
        if p_y.shape[0] != self._cond_x_y.shape[1]:
            raise IndexError(
                'col of p_y({0}) should be the same as col of cond_x_y({1})'.format(
                    p_y.shape[0], self._cond_x_y.shape[1]))

        p_y_x = []
        indices = []
        for y in list(p_y.index):
            # p(x|y)p(y)
            p_y_x.append(self._cond_x_y[y][x] * p_y[y])
            indices.append(y)
        p_y_x = pd.DataFrame(p_y_x).fillna(0.0)
        p_y_x.index = indices
        # p(x|y)p(y)/p(x)
        p_y_x = p_y_x / p_y_x.sum()
        return p_y_x[0]
