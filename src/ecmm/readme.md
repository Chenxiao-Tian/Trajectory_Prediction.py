# Endpoint-conditioned Markov models

## Project Summary

Using endpoint-conditioned Markov models allows the representation of origin-destination paths as higher-order networks.

### [2023-10-06] Meeting

Sumamry

- https://networks.skewed.de
- https://ogb.stanford.edu

- Clean up the two projects on github and transfer the literature to Zotero
- Do more simulations on the 2D regular random walk and Percolation Model
- Collect more research papers and summary their highlights on Central Limit Theorems or Other Subjects
### [2023-09-29 Fri] Meeting

- **Participants**: Juergen Hackl (JH), Chenxiao-Tian (CT)
-
- **Minutes**:
  - Develop Continuous Endpoint Markov Chain Model：How to deal with Function Spaces rather than finite state spaces for discrete case？
  - Central Limit Theorem on 2D Network and Potential Applications
  - Potential Collaboration Faculties
- **Tasks**
  - [ ] Refine Continuous Case Endpoint Markov Chain Theory
  - [ ] Do Some Random Walk Simulations on 2D Network and Verify Some CLT by Simulations
  - [ ] Potential Collaborations：1.ORFE： https://klusowski.princeton.edu/ （neural networks）  2.Math：https://www.math.princeton.edu/people/allan-sly（probability）3.ORFE： https://fan.princeton.edu/research （Graphical and Network modeling）4.Math：https://web.math.princeton.edu/~chang/ （geometry）5.Looking for more applications：faculty under Computer Science/Computational Neuroscience？
## Project Meetings
### [2023-09-22 Fri] Meeting

- **Participants**: Juergen Hackl (JH), Chenxiao-Tian (CT)
-
- **Minutes**:
  - Discuss about Theory Part vs Applications Part in a network science related paper: good applications and theory with tight logic(even without finding real data support) are both could be a highlight for a paper in this field.
  - Discuss other potential probability models which could be used in network science,e.g., percolation model.
  - Discuss some related electric papers/books resource on the group zotero library,potential research cooperation possibility.
- **Tasks**
  - [ ] For the theory part, develop some mathematical model and logic for the  Continuous Endpoint-conditioned Markov Chain with continuous time parameter Transition matrix
  - [ ] For the application part, we continuously are focused on data collection and coding for the discrete Endpoint-conditioned/Contiuuous Markov Chain modeled by Poisson Distribution.
### [2023-09-12 Tue] Meeting

- **Participants**: Juergen Hackl (JH), Chenxiao-Tian (CT)

- **Minutes**:

  - Discussed the initial idea of having endpoint-conditioned Markov models to better describe physical processes that follow an origin-destination assumption

  - Highlighted potential challenges associated with the discrete model where $n$ steps are given:

    1. $n$ might be shorter than the shortest path i.e., walker will stop in the middle of the path

    2. $n$ is a number which cannot reach the target node at step $n$ e.g., On an x-axis, start from the 0, odd number target node can not be reached by even number steps

  - A continuous model might be better for capturing time-related processes like road transportation

  - The model can have two flavors:

    1. When the walker reaches the destination node, the process ends

    2. The walker will continue his walk even after he visited the destination node

  - the parameters $n$ and $t$ could be interpreted as indicators of how efficient a process is. i.e., if $n$ is close to the shortest path, processes might be optimized（i.e. more restrictions and less choices and randomness for possible walking paths) while $n$ increases, more randomness can be observed. in the extreme case where $n \to \infty$ a "classical" random walker process can be observed(i.e.,no extra restriction on the random walker when n is infinite)

  - There might be a phase transition when the regime shifts from a "random walker" to a "shortest path walker".

    - We do not know how this transition would look

    - We do not know how topology influences this transition

    - The most common phase transition timepoint may happen when the timepoint is "closed" to n and t, it maybe exist other phase transition timepoint preference in real-world datasets.

- **Tasks**

  - [X] JH: [Set up a GitHub project and upload any pre-existing code.](https://github.com/cisgroup/PROJ-2023-ecmm/issues/1)
  - [ ] CT: [Refine the code to ensure its functionality.](https://github.com/cisgroup/PROJ-2023-ecmm/issues/2)
  - [ ] CT: [Conduct numerical experiments to investigate potential phase transitions.](https://github.com/cisgroup/PROJ-2023-ecmm/issues/3)
  - [ ] JH and CT: [Brainstorm and gather potential real-world datasets for validating the model.](https://github.com/cisgroup/PROJ-2023-ecmm/issues/4)
