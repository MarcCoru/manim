from manimlib.imports import *
import os


INDENT = .25*RIGHT
PHI_DOT_RADIUS = 0.05
THETA_DOT_RADIUS = 0.1
DATA_DOT_RADIUS = 0.05
datacolor = Color(hex="#92c1e9")

THETA_ANNOT_OFFSET = np.array([-.5,.5,0])
PHI_ANNOT_OFFSET  = np.array([-.3,.3,0])
STEP_SIZE = .5
MAML_INNER_STEP_SIZE = 0.6
MAML_OUTER_STEP_SIZE = 1
SHOW_ALGORITHM = False

class Indicate(Transform):
    CONFIG = {
        "rate_func": there_and_back,
        "scale_factor": 1,
        "color": datacolor,
    }

    def create_target(self):
        target = self.mobject.copy()
        target.set_opacity(1)
        return target


class Distribution():
    def __init__(self, mean, cov, id="1"):
        self.mean = mean
        self.cov = cov
        self.circle = Circle(arc_center=self.mean, radius=self.cov[0,0] * 5, stroke_width=0, color=datacolor,
                          fill_color=Color(hex="#cdcdcd"), fill_opacity=0.33)
        self.dot = Dot(self.mean, color=Color(hex="#cdcdcd"))
        self.annot = TextMobject(r"\tiny $\phi_"+id+r"^\ast$", color=Color(hex="#cdcdcd")).next_to(self.dot,np.array([.2,.2,0]))
    def sample(self, N_samples=3):
        pts = np.random.multivariate_normal(self.mean, self.cov, N_samples)
        return pts

    def play(self):
        return self.circle # VGroup(self.circle, self.dot, self.annot)

class Indicator():
    def __init__(self, init_position, opacity=1):
        self.indicator = RoundedRectangle(height=.3, width=0.1, corner_radius=0.03, color=datacolor, stroke_width=0,
        fill_color = datacolor, fill_opacity=opacity).next_to(init_position, LEFT*.5)
        self.opacity = opacity

    def move(self, target_position):
        self.indicator.generate_target()
        self.indicator.target = RoundedRectangle(height=.3, width=0.1, corner_radius=0.05, color=datacolor, stroke_width=0,
                                                 fill_color = datacolor, fill_opacity=self.opacity).next_to(target_position, LEFT * .5)
        return self.indicator


class RegularGradientDescent(Scene):

    def construct(self):
        size = r"\scriptsize "

        model_tex = TextMobject(size + r"neural network model $y = f_\phi$(x)").to_edge(TOP + LEFT)
        dist_tex = TextMobject(size + r"$p(\mathcal{D})$: distribution over data points").next_to(model_tex, DOWN).align_to(model_tex,LEFT)


        theta_tex = TextMobject(size + r"randomly initialize $\phi$").next_to(dist_tex, DOWN).align_to(dist_tex,LEFT)

        data_tex = TextMobject(size + r"sample ${D} \sim p(\mathcal{D})$").next_to(theta_tex, DOWN).align_to(theta_tex,LEFT).shift(INDENT)
        grad_tex = TextMobject(size + r"evaluate $\mathbf{g}=\nabla\mathcal{L}(f_{\phi},{D})$").next_to(data_tex, DOWN).align_to(data_tex,LEFT)
        update_tex = TextMobject(size + r"update parameters $\phi \leftarrow \phi - \alpha\mathbf{g}$").next_to(grad_tex, DOWN).align_to(
            grad_tex, LEFT)


        if SHOW_ALGORITHM:
            texlines = [model_tex, dist_tex, theta_tex, grad_tex, update_tex, data_tex]
            self.play(*[FadeIn(tex) for tex in texlines])
        indicator = Indicator(init_position=model_tex, opacity=1 if SHOW_ALGORITHM else 0)

        N_samples = 6

        theta_coords = np.array([1,0,0])
        theta_dot = Dot(theta_coords, radius=THETA_DOT_RADIUS)
        theta_annot = TextMobject(size + r"$\theta$").next_to(theta_dot, direction=THETA_ANNOT_OFFSET)
        # add a copy since theta_dot will be moved
        #self.add(Dot(theta_coords, radius=THETA_DOT_RADIUS))

        dist1 = Distribution(mean=np.array([5, 0, 0]), cov=np.eye(3) * .2)
        dist2 = Distribution(mean=np.array([4, -2, 0]), cov=np.eye(3) * .2)
        dist3 = Distribution(mean=np.array([3, 2, 0]), cov=np.eye(3) * .2)

        self.play(FadeIn(dist1.play()), FadeIn(dist2.play()),
                  FadeIn(dist3.play()), MoveToTarget(indicator.move(dist_tex)))

        self.play(FadeIn(theta_dot), FadeIn(theta_annot), MoveToTarget(indicator.move(theta_tex)))

        for i in range(3):
            pts = np.vstack([dist1.sample(N_samples),dist2.sample(N_samples),dist3.sample(N_samples)])
            dots = [Dot(pt, radius=DATA_DOT_RADIUS, color=datacolor) for pt in pts]

            self.play(
                *[FadeIn(dot) for dot in dots],
                MoveToTarget(indicator.move(data_tex))
                )

            ds = (pts - theta_coords) * step_size
            #ds /= np.linalg.norm(ds,2,axis=1)[:,None]
            arrows = [Arrow(theta_coords, d + theta_coords, stroke_width=0.5,tip_length=0.1, color=datacolor) for d in ds]
            self.play(*[GrowArrow(arrow) for arrow in arrows], MoveToTarget(indicator.move(grad_tex)))

            new_theta_coords = theta_coords + ds.mean(0)
            target_arrow = Arrow(theta_coords, new_theta_coords, stroke_width=0.75, tip_length=0.3)
            self.play(*[Transform(arrow, target_arrow) for arrow in arrows])
            #self.wait(1)

            new_dot = Dot(new_theta_coords, radius=PHI_DOT_RADIUS)
            theta_dot.generate_target(use_deepcopy=True)
            theta_dot.target = new_dot
            theta_annot.generate_target()
            theta_annot.target = TextMobject(size + r"$\theta$").next_to(theta_dot.target, direction=THETA_ANNOT_OFFSET)
            self.play(MoveToTarget(theta_dot), MoveToTarget(theta_annot), *[FadeOut(dot) for dot in dots],
                      MoveToTarget(indicator.move(update_tex)))

            theta_dot = new_dot
            theta_coords = new_theta_coords

class RegularGradientDescent_no_alg(Scene):

    def construct(self):
        size = r"\scriptsize "

        model_tex = TextMobject(size + r"neural network model $y = f_\phi$(x)").to_edge(TOP + LEFT)
        dist_tex = TextMobject(size + r"$p(\mathcal{D})$: distribution over data points").next_to(model_tex, DOWN).align_to(model_tex,LEFT)


        theta_tex = TextMobject(size + r"randomly initialize $\phi$").next_to(dist_tex, DOWN).align_to(dist_tex,LEFT)

        data_tex = TextMobject(size + r"sample ${D} \sim p(\mathcal{D})$").next_to(theta_tex, DOWN).align_to(theta_tex,LEFT).shift(INDENT)
        grad_tex = TextMobject(size + r"evaluate $\mathbf{g}=\nabla\mathcal{L}(f_{\phi},{D})$").next_to(data_tex, DOWN).align_to(data_tex,LEFT)
        update_tex = TextMobject(size + r"update parameters $\phi \leftarrow \phi - \alpha\mathbf{g}$").next_to(grad_tex, DOWN).align_to(
            grad_tex, LEFT)


        if SHOW_ALGORITHM:
            texlines = [model_tex, dist_tex, theta_tex, grad_tex, update_tex, data_tex]
            self.play(*[FadeIn(tex) for tex in texlines])
        indicator = Indicator(init_position=model_tex, opacity=1 if SHOW_ALGORITHM else 0)


        N_samples = 6
        step_size = .5

        theta_coords = np.array([1,0,0])
        theta_dot = Dot(theta_coords, radius=THETA_DOT_RADIUS)
        theta_annot = TextMobject(size + r"$\theta$").next_to(theta_dot, direction=THETA_ANNOT_OFFSET)
        # add a copy since theta_dot will be moved
        #self.add(Dot(theta_coords, radius=THETA_DOT_RADIUS))

        dist1 = Distribution(mean=np.array([5, 0, 0]), cov=np.eye(3) * .2)
        dist2 = Distribution(mean=np.array([4, -2, 0]), cov=np.eye(3) * .2)
        dist3 = Distribution(mean=np.array([3, 2, 0]), cov=np.eye(3) * .2)

        self.play(FadeIn(dist1.play()), FadeIn(dist2.play()),
                  FadeIn(dist3.play()), MoveToTarget(indicator.move(dist_tex)))

        self.play(FadeIn(theta_dot), FadeIn(theta_annot), MoveToTarget(indicator.move(theta_tex)))

        for i in range(3):
            pts = np.vstack([dist1.sample(N_samples),dist2.sample(N_samples),dist3.sample(N_samples)])
            dots = [Dot(pt, radius=DATA_DOT_RADIUS, color=datacolor) for pt in pts]

            self.play(
                *[FadeIn(dot) for dot in dots],
                MoveToTarget(indicator.move(data_tex))
                )

            ds = (pts - theta_coords) * STEP_SIZE
            #ds /= np.linalg.norm(ds,2,axis=1)[:,None]
            arrows = [Arrow(theta_coords, d + theta_coords, stroke_width=0.5,tip_length=0.1, color=datacolor) for d in ds]
            self.play(*[GrowArrow(arrow) for arrow in arrows], MoveToTarget(indicator.move(grad_tex)))

            new_theta_coords = theta_coords + ds.mean(0)
            target_arrow = Arrow(theta_coords, new_theta_coords, stroke_width=0.75, tip_length=0.3)
            self.play(*[Transform(arrow, target_arrow) for arrow in arrows])
            #self.wait(1)

            new_dot = Dot(new_theta_coords, radius=PHI_DOT_RADIUS)
            theta_dot.generate_target(use_deepcopy=True)
            theta_dot.target = new_dot
            theta_annot.generate_target()
            theta_annot.target = TextMobject(size + r"$\theta$").next_to(theta_dot.target, direction=THETA_ANNOT_OFFSET)
            self.play(MoveToTarget(theta_dot), MoveToTarget(theta_annot), *[FadeOut(dot) for dot in dots],
                      MoveToTarget(indicator.move(update_tex)))

            theta_dot = new_dot
            theta_coords = new_theta_coords


class MAML(Scene):

    def construct(self):
        size = r"\scriptsize "


        model_tex = TextMobject(size + r"neural network model $y = f_\phi$(x)").to_edge(TOP*-3.5 + LEFT)
        dist_tex = TextMobject(size + r"$p(\mathcal{T})$: distribution over tasks").next_to(model_tex, DOWN).align_to(model_tex, LEFT)
        theta_tex = TextMobject(size + r"randomly initialize $\theta$").next_to(dist_tex, DOWN).align_to(model_tex,LEFT)

        tasks_tex = TextMobject(size + r"foreach new task $\tau_i \sim p(\mathcal{T})$").next_to(theta_tex, DOWN).align_to(theta_tex, LEFT).shift(INDENT)

        #init_task_tex = TextMobject(size + r"initialize $\phi_{i}$ with $\theta$").next_to(tasks_tex, DOWN).align_to(
            #tasks_tex, LEFT).shift(RIGHT)



        supportdata_tex = TextMobject(size + r"sample ${D}_\text{support} \sim p(\tau_i)$").next_to(tasks_tex, DOWN).align_to(
            tasks_tex,LEFT).shift(INDENT)
        grad_tex = TextMobject(size + r"evaluate $\mathbf{g}=\nabla_{\phi_{i}}\mathcal{L}_{\tau_{i}}(f_{\phi_{i}},{D}_\text{support})$").next_to(
            supportdata_tex,DOWN).align_to(
            supportdata_tex, LEFT)
        update_tex = TextMobject(size + r"adapt parameters $\phi_{i} \leftarrow \theta - \alpha\mathbf{g}$").next_to(
            grad_tex, DOWN).align_to(
            grad_tex, LEFT)
        querydata_tex = TextMobject(size + r"sample ${D}_\text{query} \sim p(\tau_i)$").next_to(update_tex,
                                                                                                    DOWN).align_to(update_tex, LEFT)
        query_loss_tex = TextMobject(size + r"evaluate test loss $\mathcal{L}_{\tau_{i}}(f_{\phi_{i}},{D}_\text{query})$").next_to(
            querydata_tex, DOWN).align_to(
            querydata_tex, LEFT)


        outer_update_tex = TextMobject(size + r"update $\theta \leftarrow \theta - \beta \sum_{\tau_{i} \sim p(\tau)} \nabla_{\theta}\mathcal{L}_{\tau_{i}}(f_{\phi_{i}},{D}_\text{query}^{\tau_i})$").next_to(query_loss_tex,
                                                                                                 DOWN).align_to(
            tasks_tex, LEFT)

        if SHOW_ALGORITHM:
            texlines = [model_tex, dist_tex, theta_tex, tasks_tex, supportdata_tex, querydata_tex, grad_tex, update_tex,
                        query_loss_tex, outer_update_tex]

            self.play(*[FadeIn(tex) for tex in texlines])
        indicator = Indicator(init_position=model_tex, opacity=1 if SHOW_ALGORITHM else 0)


        N_samples = 6

        theta_coords = np.array([1,0,0])
        theta_dot = Dot(theta_coords, radius = THETA_DOT_RADIUS)
        theta_annot = TextMobject(size + r"$\theta$").next_to(theta_dot, direction=THETA_ANNOT_OFFSET)

        dist1 = Distribution(mean=np.array([5, 0, 0]), cov=np.eye(3) * .2, id="1")
        dist2 = Distribution(mean=np.array([4, -2, 0]), cov=np.eye(3) * .2, id="2")
        dist3 = Distribution(mean=np.array([3, 2, 0]), cov=np.eye(3) * .2, id="3")

        #self.play(FadeIn(dist1.play()), FadeIn(dist2.play()),
        #          FadeIn(dist3.play()), MoveToTarget(indicator.move(dist_tex)))

        #self.play(FadeIn(theta_dot), FadeIn(theta_annot), MoveToTarget(indicator.move(theta_tex)))
        self.add(dist1.play(),dist2.play(),dist3.play(),theta_dot,theta_annot)


        #self.play(, FadeIn(update_tex))

        for i in range(3):
            phi_coords = []
            dists = [dist1,dist2,dist3]

            self.play(MoveToTarget(indicator.move(tasks_tex)))
            #self.wait(0.2)
            #self.play(*[Indicate(dist.play()) for dist in dists])

            dots = []
            allsupportpts = []
            for dist in dists:
                supportpts = np.vstack([dist.sample(N_samples // 2)])
                allsupportpts.append(supportpts)
                dots += [Dot(pt, radius=DATA_DOT_RADIUS, color=datacolor) for pt in supportpts]

            self.play(*[FadeIn(dot) for dot in dots],
                MoveToTarget(indicator.move(supportdata_tex)))

            supportarrows = []
            allds = []
            for supportpts in allsupportpts:
                ds = (supportpts - theta_coords) * MAML_INNER_STEP_SIZE
                #ds /= np.linalg.norm(ds,2,axis=1)[:,None]
                supportarrows.append([Arrow(theta_coords, d + theta_coords, stroke_width=0.5,tip_length=0.1, color=datacolor) for d in ds])
                allds.append(ds)

            self.play(*[GrowArrow(arrow) for arrows_per_phi in supportarrows for arrow in arrows_per_phi],
                      MoveToTarget(indicator.move(grad_tex))
            #FadeIn(grad_tex) if i == 0 else Indicate(grad_tex, run_time=3)
            )

            target_arrows = []
            for ds in allds:
                phi_coord = theta_coords + ds.mean(0)
                phi_coords += [phi_coord]
                target_arrows += [Arrow(theta_coords, phi_coord, stroke_width=0.75,tip_length=0.2)]

            transforms = list()
            for arrows, target_arrow in zip(supportarrows, target_arrows):
                for arrow in arrows:
                    transforms.append(Transform(arrow, target_arrow))

            self.play(*transforms)
            #self.wait(1)

            phi_dots = []
            phi_annots = []
            for phi_coord, i in zip(phi_coords, [2,3,1]):
                phi_dot = Dot(theta_coords, radius = THETA_DOT_RADIUS)
                self.add(phi_dot)
                phi_dot.generate_target()
                phi_dot.target = Dot(phi_coord, radius=PHI_DOT_RADIUS)
                phi_dots += [phi_dot]
                phi_annots.append(TextMobject(r"\tiny $\phi_"+str(i)+"$").next_to(phi_dot.target,PHI_ANNOT_OFFSET))
            self.play(*[MoveToTarget(phi_dot) for phi_dot in phi_dots],
                      *[FadeIn(phi_annot) for phi_annot in phi_annots],
                      *[FadeOut(dot) for dot in dots],
                      *[FadeOut(target_arrow) for target_arrow in target_arrows],
                      *[FadeOut(arrow) for support_arrows_per_phi in supportarrows for arrow in support_arrows_per_phi],
                      MoveToTarget(indicator.move(update_tex)))

            dots = []
            allquerypts = []
            for dist in dists:
                querypts = np.vstack([dist.sample(N_samples // 2)])
                allquerypts.append(querypts)
                dots += [Dot(pt, radius=DATA_DOT_RADIUS, color=datacolor) for pt in querypts]

            self.play(
                *[FadeIn(dot) for dot in dots],
                MoveToTarget(indicator.move(querydata_tex))
                # FadeIn(data_tex) if i == 0 else Indicate(data_tex)
            )
            self.wait(0.3)

            allqueryarrows = []
            for phi_coord, querypts in zip(phi_coords, allquerypts):
                ds = (querypts - phi_coord)

                for d in ds:
                    allqueryarrows.append(Arrow(phi_coord, d + phi_coord, color=datacolor,stroke_width=0.5,tip_length=0.1))
                    allds.append(d)

            self.play(*[GrowArrow(arrow) for arrow in allqueryarrows],
                      *[FadeOut(dot) for dot in dots],
                      MoveToTarget(indicator.move(query_loss_tex)))
            self.wait(0.3)

            old_theta_coords = theta_coords
            theta_coords = theta_coords + np.vstack(allds).mean(0) * MAML_OUTER_STEP_SIZE
            target_arrow = Arrow(old_theta_coords, theta_coords, stroke_width=1,tip_length=0.2)
            self.play(*[Transform(arrow, target_arrow) for arrow in allqueryarrows],
                      *[FadeOut(dot) for dot in phi_dots],
                      *[FadeOut(phi_annot) for phi_annot in phi_annots])

            theta_dot = Dot(old_theta_coords, radius = THETA_DOT_RADIUS)
            theta_dot.generate_target()
            theta_dot.target = Dot(theta_coords, radius = THETA_DOT_RADIUS)
            theta_annot.target = TextMobject(size + r"$\theta$").next_to(theta_dot.target, direction=THETA_ANNOT_OFFSET)

            self.play(MoveToTarget(theta_dot),
                      MoveToTarget(theta_annot),
                      MoveToTarget(indicator.move(outer_update_tex)))
            self.wait(0.5)
            pass


class Test(Scene):

    def construct(self):
        step_size = .8
        dot = Dot(np.array([-3,1,0]))
        self.play(FadeIn(dot))

        dist_mean = np.array([3, 0, 0])
        dist_cov = np.eye(3) * .24
        N_samples = 3

        for i in range(3):
            theta_c = [dot.get_coord([0]), dot.get_coord([1]), dot.get_coord([2])]

            pts = np.random.multivariate_normal(dist_mean, dist_cov, N_samples)
            dots = [Dot(pt,radius=0.05) for pt in pts]
            self.play(*[FadeIn(dot) for dot in dots])

            d = (pts.mean(0) - theta_c) * step_size
            new_dot = Dot(theta_c + d)
            dot.generate_target(use_deepcopy=True)
            dot.target = new_dot
            self.play(MoveToTarget(dot))

            self.play(*[FadeOut(dot) for dot in dots])
            dot = new_dot
