#!/bin/bash
# Start munge, MariaDB, slurmdbd, slurmctld and slurmd on one node, then idle.
# slurm.conf is generated at start so the node line matches the CPUs and memory of the host.
set -euo pipefail

host=$(hostname)
echo "127.0.0.1 ${host}" >> /etc/hosts

mkdir -p /run/munge /var/log/munge /var/lib/munge
chown munge:munge /run/munge /var/log/munge /var/lib/munge
gosu munge munged
mkdir -p /run/mysqld && chown mysql:mysql /run/mysqld
gosu mysql mysqld_safe --skip-networking=0 >/var/log/mysqld_safe.log 2>&1 &
for _ in $(seq 1 60); do mysqladmin ping --silent 2>/dev/null && break; sleep 1; done
mysql -e "CREATE DATABASE IF NOT EXISTS slurm_acct_db;
          CREATE USER IF NOT EXISTS 'slurm'@'localhost' IDENTIFIED BY 'slurm';
          GRANT ALL ON slurm_acct_db.* TO 'slurm'@'localhost';"

node=$(slurmd -C | head -n1)
cat > /etc/slurm/slurm.conf <<CONF
ClusterName=linux
SlurmctldHost=${host}
SlurmUser=slurm
AuthType=auth/munge
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldPidFile=/run/slurm/slurmctld.pid
SlurmdPidFile=/run/slurm/slurmd.pid
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
ProctrackType=proctrack/linuxproc
TaskPlugin=task/none
SelectType=select/cons_tres
SelectTypeParameters=CR_CPU
SchedulerType=sched/backfill
ReturnToService=2
MpiDefault=none
JobAcctGatherType=jobacct_gather/linux
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=localhost
${node}
PartitionName=normal Nodes=ALL Default=YES MaxTime=INFINITE State=UP
CONF
echo CgroupPlugin=disabled > /etc/slurm/cgroup.conf
chown slurm:slurm /etc/slurm/slurm.conf /etc/slurm/cgroup.conf

slurmdbd
for _ in $(seq 1 60); do sacctmgr -n show cluster >/dev/null 2>&1 && break; sleep 1; done
sacctmgr -i add cluster linux >/dev/null
slurmctld
for _ in $(seq 1 60); do sinfo -h >/dev/null 2>&1 && break; sleep 1; done
slurmd
sacctmgr -i add account none,test Cluster=linux Description=none Organization=none >/dev/null || true
sacctmgr -i add user root Account=none DefaultAccount=none >/dev/null || true

tail -F /var/log/slurm/*.log
