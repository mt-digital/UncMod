using Distributed

# Launch worker processes.
num_cores = parse(Int, ENV["SLURM_CPUS_PER_TASK"])

addprocs(num_cores)

println("Number of cores: ", nprocs())
println("Number of workers: ", nworkers())

# Each worker gets its own id and hostname.
for i in workers()
    id, pid, host = fetch(@spawnat i (myid(), getpid(), gethostname()))
    println(id, " ", pid, " ", host)
end

# Clean up, remove workers.
for i in workers()
    rmprocs(i)
end
