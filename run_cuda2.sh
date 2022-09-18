seed=0
device=2
for datasets in 'wadi'
do
  for epoch in 200
  do
    for batch in 64
    do
      for n_critics in 5 # 3 4 5
      do
        for lr in 1e-4 1e-5 1e-6
        do
          for latent_space_dim in 10 20 30
          do
            for beta1 in 0.5
            do
              for beta2 in 0.999
                do
                  python3 -u main.py --seed $seed --device $device --datasets $datasets --epoch $epoch --batch $batch --n_critics $n_critics --lr $lr --latent_space_dim $latent_space_dim --beta1 $beta1 --beta2 $beta2
                done
            done
          done
        done
      done
    done
  done
done
