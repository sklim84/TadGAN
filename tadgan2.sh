mode=train
seed=0
device=2
for datasets in 'wadi'  # wadi, swat
do
  for epoch in 200
  do
    for batch in 1
    do
      for seq_len in 12
      do
        for n_critics in 5
        do
          for lr in 1e-5 1e-6 1e-7
          do
            for latent_space_dim in 40 # 20 30 40
            do
              for beta1 in 0.5
              do
                for beta2 in 0.999
                  do
                    python3 -u main.py --tadgan tadgan --mode $mode --seed $seed --device $device --datasets $datasets --epoch $epoch --batch $batch --seq_len $seq_len --n_critics $n_critics --lr $lr --latent_space_dim $latent_space_dim --beta1 $beta1 --beta2 $beta2
                  done
              done
            done
          done
        done
      done
    done
  done
done
