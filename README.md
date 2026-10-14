&#x20;<div align="center">



# NumPy Mastery



\### \*The array foundation underneath machine learning and data science.\*



\*\*Python 3.10+\*\* · \*\*NumPy\*\* · \*\*Jupyter Notebooks\*\* · \*\*Focus: Machine Learning and Data Science\*\*



\*\*11 topics\*\* · \*\*2 phases\*\* · \*\*19 notebooks\*\* · \*\*15 PDF notes\*\* · \*\*1 applied project\*\*



</div>



\---



\## Why this repository exists



Look underneath many of Python's data science and machine learning tools and you'll find arrays. NumPy provides the numerical foundation for working with them. Pandas builds on NumPy's array capabilities, Scikit-learn works extensively with array-like data, and the computations behind many machine learning algorithms involve vector and matrix operations.



I'm working toward a career as a \*\*Machine Learning Engineer and Data Scientist\*\*, so I didn't treat NumPy as a quick stop on the way to the "real" material. I treated it as a foundation worth understanding properly. This repository documents that process: learning how arrays behave, how their dimensions affect calculations, and how to express numerical operations without relying unnecessarily on Python loops.



The material progresses from array fundamentals to mathematical operations, aggregation, and linear algebra. Concept notebooks establish the concepts, practice notebooks reinforce them through independent problem-solving, and supporting PDF resources provide additional explanations and questions for revision.



> The goal was never to memorize function names. It was to understand \*\*why\*\* an operation exists, \*\*when\*\* to use it, and \*\*what can go wrong\*\* when its behavior is misunderstood.



\*\*Author:\*\* \[Abdullah Waseem](https://github.com/TheAbdullahWaseem)



\---



\## Contents



\- \[The learning path](#the-learning-path)

\- \[The curriculum](#the-curriculum)

\- \[How NumPy connects to machine learning](#how-numpy-connects-to-machine-learning)

\- \[Put to work: Superstore Sales Analysis](#put-to-work-superstore-sales-analysis)

\- \[Three ideas this repository keeps coming back to](#three-ideas-this-repository-keeps-coming-back-to)

\- \[How the repository is organized](#how-the-repository-is-organized)

\- \[Getting started](#getting-started)

\- \[Where this leads](#where-this-leads)

\- \[About me](#about-me)



\---



\## The learning path



The topics are numbered and build on each other. \*\*Phase 1\*\* focuses on understanding arrays and manipulating their structure. \*\*Phase 2\*\* moves toward numerical calculations, statistical summaries, and linear algebra.



```mermaid

flowchart LR

&#x20;   subgraph P1\["Phase 1: Core Foundations"]

&#x20;       direction LR

&#x20;       A\["01 Basics"] --> B\["02 Creation"] --> C\["03 Indexing"] --> D\["04 Copy vs View"]

&#x20;       D --> E\["05 Shape and Reshape"] --> F\["06 Iteration"] --> G\["07 Joining and Splitting"] --> H\["08 Search, Sort and Filter"]

&#x20;   end



&#x20;   subgraph P2\["Phase 2: Mathematical and Statistical Power"]

&#x20;       direction LR

&#x20;       I\["09 Mathematical Operations"] --> J\["10 Aggregation"] --> K\["11 Linear Algebra"]

&#x20;   end



&#x20;   H --> I

&#x20;   K --> L\["Machine Learning"]

```



\---



\## The curriculum



Each topic below links directly to its Jupyter notebook on GitHub. The PDF Notes column identifies the supporting materials available for that topic without linking to individual PDF files.



\### Phase 1 · Core Foundations



\*How arrays are created, accessed, reshaped, and combined.\*



| No. | Topic | What it builds | Notebook | PDF Notes |

|---|---|---|---|---|

| \*\*01\*\* | \*\*NumPy Basics\*\* | The `ndarray`, data types, attributes, and how NumPy arrays differ from Python lists | \[Open notebook →](./PHASE\_1\_Core\_Foundations/01\_NumPy\_Basics/01\_NumPy\_Basics.ipynb) | Explanation, Questions Answered |

| \*\*02\*\* | \*\*Array Creation\*\* | Building arrays from existing data, ranges, and predefined patterns | \[Open notebook →](./PHASE\_1\_Core\_Foundations/02\_Array\_Creation/02\_Array\_Creation.ipynb) | Explanation, Questions Answered |

| \*\*03\*\* | \*\*Indexing and Slicing\*\* | Selecting specific elements and subarrays across one or more dimensions | \[Open notebook →](./PHASE\_1\_Core\_Foundations/03\_Indexing\_and\_Slicing/03\_Indexing\_and\_Slicing.ipynb) | Notes |

| \*\*04\*\* | \*\*Copy vs View\*\* | Understanding shared memory, independent copies, and unintended modifications | \[Open notebook →](./PHASE\_1\_Core\_Foundations/04\_Copy\_vs\_View/04\_Copy\_vs\_View.ipynb) | Notes |

| \*\*05\*\* | \*\*Shape and Reshape\*\* | Working with dimensions, reshaping arrays, flattening, and adding axes | \[Open notebook →](./PHASE\_1\_Core\_Foundations/05\_Shape\_and\_Reshape/05\_Shape\_and\_Reshape.ipynb) | Notes |

| \*\*06\*\* | \*\*Iteration\*\* | Traversing arrays and understanding when vectorized operations are preferable | \[Open notebook →](./PHASE\_1\_Core\_Foundations/06\_Iteration/06\_Iterating\_The\_Right\_Way.ipynb) | Explanation, Problems |

| \*\*07\*\* | \*\*Joining and Splitting\*\* | Concatenating, stacking, and dividing arrays along different axes | \[Open notebook →](./PHASE\_1\_Core\_Foundations/07\_Joining\_and\_Splitting/07\_Joining\_and\_Splitting.ipynb) | Notes |

| \*\*08\*\* | \*\*Searching, Sorting, and Filtering\*\* | Locating values, ordering arrays, and selecting data with conditions | \[Open notebook →](./PHASE\_1\_Core\_Foundations/08\_Searching\_Sorting\_Filtering/08\_Searching\_Sorting\_and\_Filtering.ipynb) | Notes, Important Questions |



\### Phase 2 · Mathematical and Statistical Power



\*Turning arrays into numerical results through element-wise operations, aggregation, and linear algebra.\*



| No. | Topic | What it builds | Notebook | PDF Notes |

|---|---|---|---|---|

| \*\*09\*\* | \*\*Mathematical Operations\*\* | Element-wise arithmetic and universal functions (ufuncs) | \[Open notebook →](./PHASE\_2\_Mathematical\_and\_Statistical\_Power/09\_Mathematical\_Operations/09\_Mathematical\_Operations.ipynb) | Explanation |

| \*\*10\*\* | \*\*Aggregation Functions\*\* | Sums, means, minimums, maximums, and controlling reductions with `axis` | \[Open notebook →](./PHASE\_2\_Mathematical\_and\_Statistical\_Power/10\_Aggregation\_Functions/10\_Aggregation\_Functions.ipynb) | Explanation, Advanced Aggregation |

| \*\*11\*\* | \*\*Linear Algebra\*\* | Matrix and vector operations used in numerical computing and machine learning | \[Open notebook →](./PHASE\_2\_Mathematical\_and\_Statistical\_Power/11\_Linear\_Algebra/11\_Linear\_Algebra.ipynb) | None |



\---



\## How NumPy connects to machine learning



The value of learning NumPy becomes clearer when its concepts are connected to the problems they help solve.



| NumPy skill | Where it shows up in machine learning |

|---|---|

| \*\*Shapes and reshaping\*\* | Preparing input arrays with the dimensions expected by a model |

| \*\*Indexing and Boolean masks\*\* | Selecting observations, filtering values, and working with subsets of data |

| \*\*Vectorized mathematics\*\* | Computing predictions and numerical transformations across many observations |

| \*\*Aggregation with `axis`\*\* | Calculating feature-wise means, sums, and standard deviations |

| \*\*Copy vs view\*\* | Avoiding unintended modifications to arrays used elsewhere in a workflow |

| \*\*Linear algebra\*\* | Representing model parameters and performing calculations used by linear models |



Consider the prediction step of a simple linear model:



```python

import numpy as np



X = np.array(\[

&#x20;   \[1, 2],

&#x20;   \[3, 4]

])  # Two samples, two features



w = np.array(\[0.5, 2.0])  # One weight per feature



predictions = X @ w



print(predictions)

\# \[4.5 9.5]

```



The `@` operator performs matrix multiplication. Each row of `X` is combined with the corresponding weights in `w` to produce one prediction per sample.



This is a small example, but the underlying idea scales to larger datasets and more complex numerical models.



\---



\## Put to work: Superstore Sales Analysis



Learning a library is one thing. Applying its concepts to a real dataset is another.



I applied these foundations in my \*\*Superstore Sales Analysis\*\* project, which examines retail orders, customers, products, regions, sales, and profit. NumPy concepts helped me reason about the numerical operations used throughout the analysis, while Pandas provided the higher-level tools for cleaning, transforming, and summarizing the tabular data.



Examples of concepts that transfer directly include:



\- \*\*Vectorized operations\*\* for numerical calculations across many records.

\- \*\*Boolean masks\*\* for selecting observations that meet specific conditions.

\- \*\*Aggregation\*\* for summarizing numerical values across groups and dimensions.

\- \*\*Shape awareness\*\* for understanding the structure of arrays and the results of operations.

\- \*\*Careful data handling\*\* to avoid unintended changes when transforming data.



The project connected foundational programming concepts with a practical analytical workflow: preparing data, investigating patterns, comparing business performance, and interpreting results.



\*\*Project repository:\*\* The Superstore Sales Analysis project is being prepared for publication. Its GitHub link will be added here when available.



\---



\## Three ideas this repository keeps coming back to



NumPy offers a large collection of functions, but understanding a few fundamental ideas makes many of its operations easier to reason about. These are also common sources of mistakes when working with arrays.



\### 1. Know whether you have a \*view\* or a \*copy\*



A view shares the underlying data with its original array, while a copy has independent data. Basic slicing typically produces a view, whereas advanced indexing with integer or Boolean arrays produces a copy.



```python

import numpy as np



a = np.arange(10)



b = a\[2:5]       # Basic slicing: typically a view

b\[0] = 99



print(a\[2])      # 99: the original array changed



c = a\[\[2, 3, 4]] # Advanced indexing: a copy

c\[0] = -1



print(a\[2])      # Still 99: the copy is independent

```



Understanding this distinction helps prevent changes from affecting data you intended to preserve.



\### 2. Think in \*shapes\*



Before performing an operation, ask two questions:



\- What is the shape of the input?

\- What shape should the result have?



For example, a two-dimensional array with shape `(100, 5)` contains 100 rows and five columns. Summing along `axis=0` produces five column totals, while summing along `axis=1` produces 100 row totals.



Shape awareness is particularly important when combining arrays, reshaping data, and preparing inputs for machine learning models.



\### 3. Prefer whole-array operations when appropriate



NumPy supports vectorized operations that apply calculations across arrays without requiring an explicit Python loop.



```python

import numpy as np



prices = np.array(\[120.0, 80.0, 45.5, 300.0])



\# Apply an 18% increase to every value.

with\_tax = prices \* 1.18



print(with\_tax)

```



This expresses the operation directly and avoids manually iterating over each value. NumPy can often perform such numerical operations efficiently using compiled code.



Vectorization is not a guarantee that every expression will be faster in every situation, but it is an important approach for many numerical workloads.



Aggregation follows the same array-oriented thinking:



```python

import numpy as np



X = np.array(\[

&#x20;   \[1, 2, 3],

&#x20;   \[4, 5, 6]

])



print(X.mean(axis=0))  # One mean per column

\# \[2.5 3.5 4.5]



print(X.mean(axis=1))  # One mean per row

\# \[2. 5.]

```



The `axis` argument determines which dimension is reduced. Understanding that behavior is more useful than memorizing isolated examples.



\---



\## How the repository is organized



Each topic has its own directory containing the main lesson notebook and, where available, practice notebooks and PDF resources.



```text

NumPy-Mastery/

├── .gitignore

├── PHASE\_1\_Core\_Foundations/

│   ├── 01\_NumPy\_Basics/

│   ├── 02\_Array\_Creation/

│   ├── 03\_Indexing\_and\_Slicing/

│   ├── 04\_Copy\_vs\_View/

│   ├── 05\_Shape\_and\_Reshape/

│   ├── 06\_Iteration/

│   ├── 07\_Joining\_and\_Splitting/

│   └── 08\_Searching\_Sorting\_Filtering/

└── PHASE\_2\_Mathematical\_and\_Statistical\_Power/

&#x20;   ├── 09\_Mathematical\_Operations/

&#x20;   ├── 10\_Aggregation\_Functions/

&#x20;   └── 11\_Linear\_Algebra/

```



The structure separates the core foundations from the mathematical topics, making it easier to follow the material in sequence or revisit a particular concept during revision.



\## Getting started



\*\*Requirements:\*\* Python 3.10 or newer, NumPy, and Jupyter Notebook. Some notebooks may use additional libraries, such as Matplotlib.



\### Read on GitHub



Use the \*\*Open notebook →\*\* links in the curriculum tables. GitHub renders Jupyter notebooks in the browser, so you can read the material without installing anything.



\### Run in Google Colab



1\. Open \[Google Colab](https://colab.research.google.com/).

2\. Select \*\*File → Open notebook → GitHub\*\*.

3\. Enter `https://github.com/TheAbdullahWaseem/NumPy-Mastery`.

4\. Choose a notebook and run its cells.



\### Run locally



Clone the repository and install the required packages:



```bash

git clone https://github.com/TheAbdullahWaseem/NumPy-Mastery.git

cd NumPy-Mastery

pip install numpy jupyter

jupyter notebook

```



You can also use \[JupyterLab](https://jupyter.org/), which can be installed with `pip install jupyterlab` and launched with `jupyter lab`.



\*\*A useful way to study:\*\* Read a lesson, close it, and try to reproduce the key ideas from memory. Then solve the corresponding practice problems without immediately checking the examples. Understanding becomes more reliable when you can apply a concept independently.



\---



\## Where this leads



NumPy is one layer of a larger data science and machine learning toolkit. It does not replace the statistics, algorithms, or domain knowledge required later, but understanding numerical arrays makes those topics easier to approach.



My learning path includes:



\- \[x] \*\*NumPy\*\* — core arrays, indexing, reshaping, numerical operations, aggregation, and introductory linear algebra.

\- \[x] \*\*Pandas\*\* — data cleaning, transformation, and analysis. \[Explore my Pandas repository](https://github.com/TheAbdullahWaseem/Pandas).

\- \[x] \*\*Python and SQL\*\* — programming fundamentals and querying relational data. \[Explore my Python repository](https://github.com/TheAbdullahWaseem/Python) · \[Explore my SQL repository](https://github.com/TheAbdullahWaseem/SQL-Learning).

\- \[ ] \*\*Further NumPy practice\*\* — deepen my work with broadcasting, random number generation, and numerical implementations from scratch.

\- \[ ] \*\*Machine learning with Scikit-learn\*\* — preprocessing, modeling, and evaluation.

\- \[ ] \*\*End-to-end machine learning projects\*\* — apply the concepts in complete portfolio projects.



The objective is to keep strengthening the foundations while progressing toward increasingly realistic data science and machine learning problems.



\---



\## About me



I'm \*\*Abdullah Waseem\*\*, a Computer Science graduate working toward a career as a \*\*Machine Learning Engineer and Data Scientist\*\*. My toolkit includes Python, NumPy, Pandas, Matplotlib, Seaborn, SQL, and Power BI.



I learn best by understanding the concepts, practicing independently, and applying what I learn to meaningful problems. This repository reflects that approach: building a numerical foundation carefully rather than treating library functions as a checklist to memorize.



If you're hiring, collaborating, or learning NumPy yourself, feel free to explore my work.



\*\*GitHub:\*\* \[TheAbdullahWaseem](https://github.com/TheAbdullahWaseem)



\*If this repository helped you, a star is always appreciated.\*



