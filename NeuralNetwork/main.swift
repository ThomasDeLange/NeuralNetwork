//
//  main.swift
//  NeuralNetwork
//
//  Created by Thomas De Lange on 07-03-18.
//  Copyright © 2018 Thomas De Lange. All rights reserved.
//

import Foundation

print("Hello, World!")

var input = [
    1 : [0.0, 0.0],
    2 : [0.0, 0.1],
    3 : [1.0, 0.0],
    4 : [1.0, 1.0],
    

]

var target = [
    1 : [0.0],
    2 : [1.0],
    3 : [1.0],
    4 : [0.0]
]


let nn = NeuralNetwork(input: 2, hidden: 12, output: 1)

for _ in 1...8000{

    let randomRow = Int.random(min: 1, max: 4)
    nn.train(inputArray: input[randomRow]! , targetArray: target[randomRow]! )

}

print("\(nn.guess(inputArray: [0.0, 0.0])) has to be 0")
print("\(nn.guess(inputArray: [0.0, 0.1])) has to be 1")
print("\(nn.guess(inputArray: [1.0, 0.0])) has to be 1")
print("\(nn.guess(inputArray: [1.0, 1.0])) has to be 0")

 
