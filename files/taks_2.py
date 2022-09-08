import typing 

class BMI_Calculator(object):
    """
    This is the program that computes Body Mass Index of the member of Gym  based on the parameters below.
    
    Parameters
    ----------
    first_name : string (str)
    last_name: string (str)
    height: float (in centimeters)
    weight: float (in Kg)
    age: integer (int)
    """
    
    members_list = []
    
    def __init__(self, first_name: int, last_name: int, height: float, weight: float, age: int) -> None:
        self.first_name = "Hasan" if first_name == "Hakan" else first_name # ya birden çok Hakan girilirse? Spor salonunun sahibinin soyadına sahip olursak daha iyi olur!
        self.last_name = last_name
        self.full_name = first_name + ' ' + last_name
        self.height = height
        self.weight = weight + 5 if first_name == "Hakan" else weight
        self.age = age
        BMI_Calculator.members_list.append(self)
        
    def __str__(self):
        return f"{self.first_name}, {self.age} yaşında, {self.height} boyda, {self.weight} kiloda."    

        #return f"{self.first_name}, {self.age} yaşında, {self.height} boyda, {self.weight} kiloda. Vücut kitle indeksi: {self.bmi}. Obezlik durumu: {self.state}."    
    
    def obesity_status(self):
        self.bmi = self.weight / ( (self.height/100)**2)
        
        if (self.bmi < 18.5):
            self.state = 'Zayıf'
            
        elif (self.bmi >= 18.5 and self.bmi <= 24.9):
            self.state = 'Sağlıklı'
        
        elif (self.bmi >= 25 and self.bmi <= 29.9):
            self.state = 'Kilolu'
            
        elif (self.bmi >= 30):
            self.state = 'Obez'
        return self.state
    
    
    # Hakan'ın ağırlığını güncellersek acaba +5 ekleyecek miyiz?
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'first_name' or key == 'last_name':
                raise AttributeError('This attribute can only be set once in the init method')
            setattr(self, key, value)
        self.obesity_status()
    
    @classmethod
    def show(cls):
        hakan_item = [member for member in cls.members_list if member.first_name == 'Hasan'][0] # tek bir Hakan girildiği varsayımı altında
        cls.members_list.remove(hakan_item)
        cls.members_list.sort(key=lambda x: x.age, reverse=False)
        cls.members_list.append(hakan_item)
        for index, member in enumerate(cls.members_list):
            print(str(index + 1) + str ('-'), member)
    
    @classmethod
    def remove(cls, index):
        BMI_Calculator.members_list.pop(index - 1)
        for index, member in enumerate(BMI_Calculator.members_list):
            print(str(index + 1) + str ('-'), member)
