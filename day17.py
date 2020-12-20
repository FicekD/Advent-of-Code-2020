import os
import numpy as np


def read_data(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    lines = [line.rstrip('\n') for line in lines]
    return lines


def check_dims(states):
    x, y, z = [0, 0], [0, 0], [0, 0]
    if np.sum(states[0, :, :]) > 0:
        z[0] = 1
    if np.sum(states[-1, :, :]) > 0:
        z[1] = 1
    if np.sum(states[:, 0, :]) > 0:
        y[0] = 1
    if np.sum(states[:, -1, :]) > 0:
        y[1] = 1
    if np.sum(states[:, :, 0]) > 0:
        x[0] = 1
    if np.sum(states[:, :, -1]) > 0:
        x[1] = 1
    size = states.shape
    new_states = np.zeros(np.add(size, [sum(z), sum(y), sum(x)]), dtype=states.dtype)
    new_states[z[0]:size[0]+z[0], y[0]:size[1]+y[0], x[0]:size[2]+x[0]] = states
    return new_states


def main():
    base_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(base_path, 'data', 'day17_data.txt')
    data = read_data(data_path)
    
    states = list()
    for line in data:
        states.append([0 if char == '.' else 1 for char in line])
    states = np.array(states)
    states = np.expand_dims(states, axis=0)

    cycles = 6
    for _ in range(cycles):
        states = check_dims(states)
        print(f'{states.shape} : {np.sum(states)}')
        new_states = states.copy()
        for z in range(states.shape[0]):
            for row in range(states.shape[1]):
                for col in range(states.shape[2]):
                    area = states[z-(z>0):z+(2*(z<states.shape[0])),
                                  row-(row>0):z+(2*(row<states.shape[1])), 
                                  col-(col>0):col+(2*(col<states.shape[2]))]
                    state = states[z, row, col]
                    active = np.sum(area) - state
                    if state and not (2 <= active <= 3):
                        new_states[z, row, col] = 0
                    elif not state and active == 3:
                        new_states[z, row, col] = 1
        states = new_states
    print(np.sum(states))


if __name__ == '__main__':
    main()
