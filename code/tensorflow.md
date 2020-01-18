# TensorFlow

TensorFlow is by far the most popular deep learning framework to date. It is the preferred choice in production and it is also still widely used in research. TensorFlow itself is actually written in C++ (and CUDA) but there are various APIs that allow you to call those functions with other languages like Python, Swift, Go and JavaScript. 

---
### My Favourite Resources

These are my favourite resources. I've gone through almost all of them and I found that they did the best at explaining how to use TensorFlow. 

!> **Remember**, I am coming at this from a **researchers** perspective. So I am biased and the resources I've chosen assumes some prior knowledge about **Python** programming and **Deep learning** in general. I will list some resources

---
#### Francois Chollet

The best resource I have found is from the founder of keras ([Francois Chollet](https://fchollet.com/)). He is a very outspoken individual who is very proud of keras and how it has changed the community. He also likes to make comparisons between frameworks but overall he is very passionate about his work. He is also super active on [twitter](https://twitter.com/fchollet?lang=en) and has some interesting opinions from time to time.

The first tutorial is basically a notebook on using `tf.keras` from a deep learning perspective. I think he breaks it down quite nicely and goes through all of the important aspects that a researcher should know when using TensorFlow. If you are already familiar with TensorFlow I think you'll find almost every major point he makes useful (e.g. `Callbacks`, ) when you construct your neural networks. If you still don't see it after going through it, don't worry, it will come up.

* TensorFlow 2.0 + Keras Overview for Deep Learning Researchers - [Colab Notebook](https://colab.research.google.com/drive/1UCJt8EYjlzCs1H1d1X0iDGYJsHKwu-NO#scrollTo=zoDjozMFREDU)
* `tf.keras` for Researchers: Crash Course - [Colab Notebook](https://colab.research.google.com/drive/17u-pRZJnKN0gO5XZmq8n5A2bKGrfKEUg) 
* Inside TensorFlow: `tf.keras` - [Video 1](https://youtu.be/UYRBHFAvLSs) | [Video 2](https://www.youtube.com/watch?v=uhzGTijaw8A)


---
#### TensorFlow Website

There are a few really good tutorials on the TF website that give a really good overview of changes from TF 1.X TF 2.X as well as some more in-depth tutorials. I found the tutorials a bit difficult to navigate as there is a lot of repetition with the 'nuggets of wisdom' scattered everywhere. In addition, I find that the organization isn't really done based on the users level of knowledge. So I tried to outline how I think one should approach the tutorials based on three criteria: **absolute beginner**, **Keras users**, and **PyTorch users** which is my way of saying **Beginner**, **Intermediate** and **Advanced**.

---
**1 Absolute Beginners**

Honestly, if you're just starting out with deep learning then you should probably just dive into it using a project or take your time and go through a course and/or book. I've listed my favourite books in the next section if you're interested but I will recommend this course which is sponsored by TensorFlow. I went through the first few lectures but I got a bit bored because I had already learned this stuff. But I like the balance of explanations and code.

* Introduction to TensorFlow for Deep Learning - [Udacity](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187)

If you have a bit more free time and dedication, I would recommend you go through the TensorFlow curriculum. They break the parts necessary for learning Deep learning with TF as your platform. It has books and video courses and I personally think it is organized very well. The course I listed above is included in the curriculum.

* TensorFlow Curriculums - [Learn ML](https://www.tensorflow.org/resources/learn-ml)

---
**2 Keras Users** (Intermediate)

Most people who apply DL models to data will be in this category. They will either want simple models with fairly straightforward setups or highly complex networks. In addition, they also may have complex training regimes. They all fall into this category and from this section, you should be able to get started.

* [Keras Overview](https://www.tensorflow.org/guide/keras/overview)
  > This is a fairly long tutorial that goes through keras from top to bottom. I wouldn't recommend reading the whole thing in one go as it is a bit overwhelming. If you want to do simple models, then only look at [part 1](https://www.tensorflow.org/guide/keras/overview#build_a_simple_model). And then for a quick overview of complex models, check out [part 2](https://www.tensorflow.org/guide/keras/overview#build_complex_models) 
* [Keras Functional API](https://www.tensorflow.org/guide/keras/functional)
  > I imagine most people who are experimenting with complex models with inputs and outputs in various locations will be here.
* [Train and Evaluate with Keras](https://www.tensorflow.org/guide/keras/train_and_evaluate)
  > This is another long guide that goes through how one can train your DL model. If you're not interested in too much training customization, then you'll probably mostly interest in [part 1](https://www.tensorflow.org/guide/keras/train_and_evaluate#part_i_using_build-in_training_evaluation_loops) where you use the built-in training module.  [Part 2](https://www.tensorflow.org/guide/keras/train_and_evaluate#part_ii_writing_your_own_training_evaluation_loops_from_scratch) does things from scratch.

!> **MATLAB Users**: Although I recommend you start in the absolute beginner section to get accustomed to Python, you will probably fall into this category as well. The `Sequential` API is very similar to the new DL toolbox in the latest versions of MATLAB. Unfortunately there is no GUI yet though...

---
**3 PyTorch Users** (Advanced)

All my PyTorch and advanced Python users (me included) start here. You should feel right at home with TensorFlow using the *subclassing*. The distinction between `Layer` and `Model` is quite blurry but it's similar to the PyTorch `nn.Module`. In the end, it's super Pythonic so we should feel right at home. Finally...

* [TF for Experts](https://www.tensorflow.org/tutorials/quickstart/advanced)
  > The is a 2-minute introduction to the language. If you are familiar with PyTorch then you will find this very familiar territory. It really highlights how the two packages converged.
* [Writing Custom Layers and Models with Keras](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
  > This next tutorial will go into more detail about the *subclassing* and how to build layers from scratch. It's very similar to PyTorch but there are a few more nuggets and subtleties that are unique to TensorFlow.
* [Train and Evaluate with Keras](https://www.tensorflow.org/guide/keras/train_and_evaluate)
  > This is another long guide that goes through how one can train your DL model. Pay special attention to [part 2](https://www.tensorflow.org/guide/keras/train_and_evaluate#part_ii_writing_your_own_training_evaluation_loops_from_scratch) where you build things from scratch as this is most similar to the PyTorch methods.
  

---
### Books

This is a bit old school but there have recently been a lot of good books released in the past 2 years that do a very good job at teaching you Machine learning (including Deep learning) from a programming perspective. They don't skip out on all of the theory but you won't find al of the derivations necessary to write the algorithms from scratch. A nice thing is that most of the books below have code on their github accounts that you can download and run yourself. 

* Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow - Aurelien Geron (2019) - [Book](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646) | [Github]()
  > This is a best selling book and I found it to be the best resources for getting a really good overview of ML with Python in general as well as a really extensive section on TF2.0 and keras.
* Deep Learning with Python - Francois Chollet (2018) - [Book](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438)
  > By the creator of Keras himself. It's a great book that goes step-by-step with lots of examples and lots of explanations.
* Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow 2 - Raschka & Mirjalili (2019) - [Book](https://www.amazon.com/Python-Machine-Learning-scikit-learn-TensorFlow/dp/1789955750/ref=sr_1_1?keywords=Python+Machine+learning&qid=1579273871&s=books&sr=1-1) | [Github]()
  > Another great book that talks about DL as well as ML in general. 


