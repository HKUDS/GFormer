<body marginheight="0"><h1>Masked Graph Transformer for Recommendation</h1>
<p>This repository contains pyTorch code and datasets for the paper:

</p>
<blockquote>
<p>Chaoliu li,  Chao Huang, Lianghao Xia, Xubin Ren, Yaowen Ye, and Yong Xu (2023)
</p>
<h2>Introduction</h2>
<p>Contrastive methods currently havve become the most successful graph-based collaborative filtering models, which follow the principle of mutual information maximization, to augment recommender systems with self-supervised information. Essentially, high-quality
data augmentation with revelant contrastive pretext tasks are necessary for performace improvement. However, to our best knowledge, most existing contrastive models hight rely on the handcrafted design of contrastive views for augmentation, which may
result in the loss of important sementice knowledge or mislead the self-supervision process with noisy signals. In this work, we aim to 
explore the augmentdation mechanism without the heuristic-based contrastive view generation by answering the question: What information should we preserve as important self-supervision signals?
we propose Rationale-aware Masked Graph Transformer(RMGT) which offers parameterized collaborative rationale discovery to distill informative user-item interaction patterns for selective augmentation.


</p>
</blockquote>
<h2>Environment</h2>
<p>The codes of HCCF are implemented and tested under the following development environment:


</p>
<p>PyTorch:

</p>
<ul>
<li>python=3.8.13</li>
<li>torch=1.9.1</li>
<li>numpy=1.19.2</li>
<li>scipy=1.9.0</li>
<li>networkx = 2.8.6</li>
</ul>
<h2>Datasets</h2>
<p>We utilized three datasets to evaluate : <i>Yelp, Ifashion, </i>and <i>LastFm</i>. Following the common settings of implicit feedback, if user $u_i$ has rated item $v_j$, then the element $(u_i, v_j)$ is set as 1, otherwise 0. We filtered out users and items with too few interactions. The datasets are divided into training set, validation set and testing set by 70:5:25


</p>
<h2>How to Run the Code</h2>
<p>Please unzip the datasets first. Also you need to create the <code>History/</code> and the <code>Models/</code> directories. The command to train RMGT on the Yelp/Ifashion/Lastfm dataset is as follows. The commands specify the hyperparameter settings that generate the reported results in the paper.

</p>
<ul>
<li>Yelp<pre><code>python Main.py --data yelp --reg 1e-4 --ssl_reg 1 --gcn 3 --ctra 1e-3 --b2 1 --pnn 1 --keepRate 0.9</code></pre>
</li>
<li>MovieLens<pre><code>python Main.py --data ifashion --reg 1e-5 --ssl_reg 1 --gcn 2 --ctra 1e-3 -- b2 1 --pnn 1 --keepRate 0.9</code></pre>
</li>
<li>Amazon<pre><code>python Main.py --data lastfm --reg 1e-4 --ssl_reg 1 --gcn 2 --ctra 1e-3 --b2 1e-6 --pnn2 --keepRate 0.9</code></pre>
</li>
</ul>
</body></html>
