class TrackerB:
    def __init__(self, nameB, trackerAA):
        self.nameB = nameB
        self.trackerAA = trackerAA

    def printB(self):
        print("insdie printB", self.nameB)
        print(self.trackerAA.printA())

class TrackerA:

    def __init__(self, nameA):
        self.nameA = nameA

    def printA(self):
        print("inside printA", self.nameA)


a = TrackerA("AA")
b = TrackerB("BB", a)

# print(a.nameA)
print(b.printB())

# x = Tracker("xx")
# y = Tracker("yy")
# print(x.ret(), x.name)
# x.noret()
# print(y.ret(), y.name)
# y.noret()

