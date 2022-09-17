#sumcore co san 16 + 48 = 54 / 96
#script 1: sequence = baseline = 1 cores, parallel 53 core
taskset -c 20  python examples/parallel.py --c "./configs/ensemble/soft.yaml" --cpu --b 64 --l 5
taskset -c 21  python examples/parallel.py --c "./configs/ensemble/soft.yaml" --cpu --e --b 64 --l 5 
taskset -c 22  python examples/parallel.py --c "./configs/ensemble/hard.yaml" --cpu --e --b 64 --l 5

taskset -c 30-83  python examples/parallel.py --c "./configs/ensemble/soft.yaml" --cpu --e --b 64 --l 5 --parallel
taskset -c 30-83  python examples/parallel.py --c "./configs/ensemble/hard.yaml" --cpu --e --b 64 --l 5 --parallel



#script 2 squence = baseline = 16 core, parallel 48 core
taskset -c 42-58  python examples/parallel.py --c "./configs/ensemble/soft.yaml" --cpu --b 64 --l 5 
taskset -c 42-58  python examples/parallel.py --c "./configs/ensemble/soft.yaml" --cpu --e --b 64 --l 5  
taskset -c 42-58  python examples/parallel.py --c "./configs/ensemble/hard.yaml" --cpu --e --b 64 --l 5 

taskset -c 42-90  python examples/parallel.py --c "./configs/ensemble/soft.yaml" --cpu --e --b 64 --l 10 --parallel
taskset -c 42-90  python examples/parallel.py --c "./configs/ensemble/hard.yaml" --cpu --e --b 64 --l 10 --parallel

