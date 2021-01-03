::# Login to AWS proxy
cd C:\Users\Kobus\
ssh -p443 -i details.pem ubuntu@ec2-13-244-167-52.af-south-1.compute.amazonaws.com

:: # Connect to WITS cluster and keep alive
ssh -o ServerAliveInterval=10 pjvanderwalt@146.141.21.100

:: # Activate anaconda
source anaconda3/bin/activate
conda activate pytorch

:: # Running Commands Directly
srun -N6 -p batch -l /bin/hostname
srun -N2 -p biggpu -l cat /proc/cpuinfo | grep model
srun -N4 -p ha -l /usr/bin/uptime

sbatch --job-name=vgg11 --partition=batch --wrap="python train_voc_vgg11.py" --output=vgg11.out
sbatch --job-name=vgg13 --partition=batch --wrap="python train_voc_vgg13.py" --output=vgg13.out
sbatch --job-name=vgg16 --partition=batch --wrap="python train_voc_vgg16.py" --output=vgg16.out

sbatch --job-name=vgg16u0 --partition=batch --wrap="python train_voc_vgg16_unfreeze_0.py" --output=vgg16u0.out
sbatch --job-name=vgg16u1 --partition=batch --wrap="python train_voc_vgg16_unfreeze_1.py" --output=vgg16u1.out
sbatch --job-name=vgg16u2 --partition=batch --wrap="python train_voc_vgg16_unfreeze_2.py" --output=vgg16u2.out
sbatch --job-name=vgg16u3 --partition=batch --wrap="python train_voc_vgg16_unfreeze_3.py" --output=vgg16u3.out

sbatch --job-name=wsgan --partition=biggpu --wrap="python train_voc_unet_adverserial.py" --output=wsgan.out

::# biggpu

::# Copy from WITS cluster to AWS proxy
scp -r pjvanderwalt@146.141.21.100:~/weakly-supervised-segmentation/models/checkpoints/ ~/
scp -r pjvanderwalt@146.141.21.100:~/weakly-supervised-segmentation/training/output/ ~/
::# Copy from AWS proxy to machine
scp -P 443 -i details.pem -r ubuntu@ec2-13-244-167-52.af-south-1.compute.amazonaws.com:~/checkpoints /
scp -P 443 -i details.pem -r ubuntu@ec2-13-244-167-52.af-south-1.compute.amazonaws.com:~/output /

:: # CHPC interactive session
qsub -I -P CSCI1340 -q serial -l select=1:ncpus=1:mpiprocs=1:nodetype=haswell_reg

:: # Copy specific weight file
:: # Step1: rsync -chavzP --stats pjvanderwalt@146.141.21.100:~/weakly-supervised-segmentation/models/checkpoints/multiclass_up.pt ./ 
:: # Step2: scp -P 443 -i details.pem -r ubuntu@ec2-13-244-167-52.af-south-1.compute.amazonaws.com:~/multiclass_up.pt ./
