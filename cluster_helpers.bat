::# Login to AWS proxy
cd C:\Users\Kobus\
ssh -p443 -i details.pem ubuntu@ec2-13-244-167-52.af-south-1.compute.amazonaws.com

:: # Connect to WITS cluster and keep alive
ssh -o ServerAliveInterval=10 pjvanderwalt@146.141.21.100

:: # Activate anaconda
source anaconda3/bin/activate

::# Copy from WITS cluster to AWS proxy
scp -r pjvanderwalt@146.141.21.100:~/weakly-supervised-segmentation/models/checkpoints/ ~/
::# Copy from AWS proxy to machine
scp -P 443 -i details.pem -r ubuntu@ec2-13-244-167-52.af-south-1.compute.amazonaws.com:~/checkpoints /

:: # CHPC interactive session
qsub -I -P CSCI1340 -q serial -l select=1:ncpus=1:mpiprocs=1:nodetype=haswell_reg

:: # Copy specific weight file
:: # Step1: rsync -chavzP --stats pjvanderwalt@146.141.21.100:~/weakly-supervised-segmentation/models/checkpoints/multiclass_up.pt ./ 
:: # Step2: scp -P 443 -i details.pem -r ubuntu@ec2-13-244-167-52.af-south-1.compute.amazonaws.com:~/multiclass_up.pt ./
