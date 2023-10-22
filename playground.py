def make_plot(HFA_select, correct_select, saveName):
     
    plt.title(br)
    HFA_c= HFA_selct[correct_select==1]
    HFA_nc= HFA_selct[correct_select==0]
    plt.plot(np.linspace(-700, 2300, HFA.shape[1]), np.mean(HFA_c, axis=0), color='tab:blue', label=f'Recalled: {HFA_c.shape[0]}')
    plt.plot(np.linspace(-700, 2300, HFA.shape[1]), np.mean(HFA_nc, axis=0), color='tab:orange', label=f'Not recalled: {HFA_nc.shape[0]}')
    plt.xticks(np.arange(-700, 2300, 350))
    plt.xlabel("Time (ms)")
    plt.ylabel("Normalized HFA power")
    plt.legend()
    plt.savefig(f'{savefigs}/{saveName}_{br}', dpi=300)
    plt.show()
    