from mb_train_ccd import *

if __name__ == '__main__':
    iter = 3000
    steps = 100
    target_x = 0.7
    target_y = 0.5
    has_continuous_action_space = True  # continuous action space; else discrete
    device = torch.device('cpu')

    if (torch.cuda.is_available()):
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    print("============================================================================================")

    use_kernel_loop = False  # calculating grid velocity with kernel loop
    use_sparse_FEM = False  # use sparse FEM solver

    batch_size = 32
    gui = ti.GUI("DiffRVO", res=(500, 500), background_color=0x112F41)


    sim = pyrvo.RVOSimulator(2,10)
    multisim = pyrvo.MultiRVOSimulator(batch_size,2,10)

    env = NavigationEnvs(batch_size, gui, sim, multisim, use_kernel_loop, use_sparse_FEM)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = PolicyNet(env, state_dim, action_dim, has_continuous_action_space,batch_size=batch_size).to(device)