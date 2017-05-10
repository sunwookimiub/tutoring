from car import Car
from truck import Truck
from motorcycle import Motorcycle

def main():
        car = Car(20000,'Honda', 'Accord', 2014, 'June')
        print car.purchase_price()
        print car.is_motorcycle()

        motorcycle = Motorcycle(2300, 'Kawasaki', 'Ninja', 2013, None)
        print motorcycle.purchase_price()
        print motorcycle.is_motorcycle()

if __name__ == "__main__":
        main()
