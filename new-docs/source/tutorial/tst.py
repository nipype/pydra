from pydra.design import python

if __name__ == "__main__":

    @python.define
    def TenToThePower(p: int) -> int:
        return 10**p

    ten_to_the_power = TenToThePower().split(p=[1, 2, 3, 4, 5])

    # Run the 5 tasks in parallel split across 3 processes
    outputs = ten_to_the_power(worker="cf", n_procs=3)

    p1, p2, p3, p4, p5 = outputs.out

    print(f"10^5 = {p5}")
