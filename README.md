HEART-PFL: Dual-Sided Framework for Personalized Federated LearningThis repository is the official implementation of HEART-PFL.Authors: Minjun Kim, Minje Kim📝 AbstractPersonalized Federated Learning (PFL) aims to deliver effective client-specific models under heterogeneous data distributions. However, existing approaches often suffer from shallow prototype alignment and brittle server-side knowledge transfer.To address these limitations, we propose HEART-PFL, a dual-sided framework designed to balance hierarchical semantics and global stability:Hierarchical Directional Alignment (HDA): A client-side mechanism that performs depth-aware alignment. It enforces cosine similarity in early layers for directional consistency and mean-squared matching in deeper layers for semantic precision, preserving personalization.Adversarial Knowledge Transfer (AKT): A server-side strategy that strengthens ensemble knowledge transfer via bidirectional, symmetric-KL distillation on both clean and adversarial proxy samples. This mitigates personalization bias and stabilizes global updates.Key Results:Implemented with lightweight adapters (1.46M parameters), HEART-PFL achieves state-of-the-art accuracy on Dirichlet non-IID partitions:CIFAR-100: 63.42%Flowers-102: 84.23%Caltech-101: 95.67%🛠️ InstallationWe recommend using Anaconda to manage the environment.# 1. Clone the repository
git clone [https://github.com/minjun-kim/HEART-PFL.git](https://github.com/minjun-kim/HEART-PFL.git)
cd HEART-PFL

# 2. Create environment
conda create -n heart-pfl python=3.8

# 3. Activate environment
conda activate heart-pfl

# 4. Install PyTorch & CUDA toolkit
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch

# 5. Install dependencies
pip install -r requirements.txt
📂 DatasetThis project uses torchvision.datasets. When you run the training script, the datasets will be automatically downloaded to your data directory.We evaluate our method on the following benchmarks:CIFAR-100Oxford 102 Flowers (Flowers-102)Caltech-101🚀 TrainTo train the model, run the shell scripts provided in the scripts directory.Hyperparameters (e.g., learning rate, batch size, partition alpha) can be modified directly inside each .sh file.CIFAR-100bash scripts/cifar100/ours.sh
Flowers-102bash scripts/flower102/ours.sh
Caltech-101bash scripts/caltech101/ours.sh
📍 CitationIf you find this work useful, please cite our paper:% Citation will be updated soon.
