
for f in jarvis_2021_*;
do
        cd $f
        echo $f;
        sbatch job.sh
        cd ..;
done;