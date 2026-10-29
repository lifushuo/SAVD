def generate_s2_type(batch_size=10, problem_size=100):
    import torch
    import random

    def generate_grid_rectangular(num_points, xlo, xhi, ylo, yhi, jitter_scale=0.02):
        # Find grid dimensions m x n closest possible for num_points
        n = max(1, int(torch.sqrt(torch.tensor(num_points, dtype=torch.float32)).item()))
        m = max(1, (num_points + n - 1) // n)
        x = torch.linspace(xlo, xhi, n)
        y = torch.linspace(ylo, yhi, m)
        xx, yy = torch.meshgrid(x, y)
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        
        # Randomly pick num_points
        coords = coords[torch.randperm(coords.shape[0])[:num_points]]
        # Jitter grid
        coords += (torch.rand_like(coords) - 0.5) * jitter_scale
        coords = torch.clamp(coords, 0.0, 1.0)
        return coords

    def generate_band_with_edge_points(num_points):
        # Place points in two horizontal (or vertical) bands and scatter some along the rectangular edge
        center_band = 0.25 + 0.5 * torch.rand(1).item()  # in [0.25,0.75]
        band_height = 0.10 + 0.08 * torch.rand(1).item()
        band2_height = 0.10 + 0.08 * torch.rand(1).item()
        nb1 = random.randint(num_points // 3, num_points // 2)
        nb2 = random.randint(num_points // 8, num_points // 4)
        nb_edge = num_points - (nb1 + nb2)

        # horizontal bands (center at y=center_band)
        xb1 = torch.rand(nb1, 1)
        yb1 = center_band + (torch.rand(nb1, 1) - 0.5) * band_height
        band1 = torch.cat([xb1, yb1], dim=1)

        xb2 = torch.rand(nb2, 1)
        yb2 = (1-center_band) + (torch.rand(nb2, 1) - 0.5) * band2_height
        band2 = torch.cat([xb2, yb2], dim=1)

        # edge points (split among all borders)
        edge_ns = [nb_edge // 4 for _ in range(4)]
        edge_ns[-1] += nb_edge - sum(edge_ns)
        edges = []

        # bottom, top
        for yi in [0.0, 1.0]:
            if edge_ns[0] > 0:
                edges.append(torch.cat([torch.rand(edge_ns[0],1), torch.full((edge_ns[0],1), yi)], dim=1))
            edge_ns = edge_ns[1:]
        # left, right
        for xi in [0.0, 1.0]:
            if edge_ns and edge_ns[0] > 0:
                edges.append(torch.cat([torch.full((edge_ns[0],1), xi), torch.rand(edge_ns[0],1)], dim=1))
            if edge_ns:
                edge_ns = edge_ns[1:]
        # Combine
        pts = [band1, band2] + edges
        pts = torch.cat(pts, dim=0)
        pts += (torch.rand_like(pts)-0.5)*0.01 # slight local jitter
        pts = torch.clamp(pts,0.0,1.0)
        idx = torch.randperm(pts.shape[0])
        pts = pts[idx][:num_points]
        return pts

    def generate_mild_lattice_patches(num_points):
        # split into several (2-4) blocks, fill each with grid or partial grid, randomly positioned within bounding rectangle
        nblocks = random.randint(2,4)
        p_alloc = torch.multinomial(torch.ones(nblocks), num_points, replacement=True)
        block_sizes = [(p_alloc == i).sum().item() for i in range(nblocks)]
        xy_boxes = []
        pts = []
        for i in range(nblocks):
            margin = 0.03+0.08*random.random()
            left = margin + (1.0-2*margin)*random.random()
            top = margin + (1.0-2*margin)*random.random()
            width = min(0.2+random.random()*0.25, 1.0-left)
            height = min(0.2+random.random()*0.20, 1.0-top)
            box_pts = max(1,block_sizes[i])
            arr = generate_grid_rectangular(box_pts, left, left+width, top, top+height, jitter_scale=0.01+random.random()*0.015)
            pts.append(arr)
        pts = torch.cat(pts, dim=0)
        pts = pts[torch.randperm(pts.shape[0])][:num_points]
        pts = torch.clamp(pts, 0.0, 1.0)
        return pts

    def generate_central_regular_grid_plus_periphery(num_points):
        # Make core block (half points), then add a ring or scatter at periphery
        ncore = num_points // 2
        nring = num_points-ncore
        g = generate_grid_rectangular(ncore,0.2,0.8,0.2,0.8,jitter_scale=0.015+0.02*random.random())
        # peripheral ring
        thetas = torch.rand(nring)*2*3.1415926535898
        rads = (torch.ones(nring)*0.5)+0.03*torch.randn(nring)
        centers = torch.tensor([0.5,0.5])
        ring = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=1) * rads.unsqueeze(1)
        ring = ring*0.8+centers # shrink and center
        ring += (torch.rand_like(ring)-0.5)*0.01
        ring = torch.clamp(ring,0.0,1.0)
        pts = torch.cat([g,ring],dim=0)
        pts = pts[torch.randperm(pts.shape[0])][:num_points]
        return pts

    def generate_vertical_or_horizontal_stripes(num_points):
        num_stripes=random.randint(2,4)
        nps = torch.multinomial(torch.ones(num_stripes), num_points, replacement=True)
        per_stripe = [(nps==i).sum().item() for i in range(num_stripes)]
        orientation = random.choice(['vertical','horizontal'])
        pts=[]
        for si in range(num_stripes):
            frac = (si+0.5)/num_stripes
            width = 0.14+0.13*random.random()
            if orientation=='vertical':
                xlo = max(0.0, frac-width/2)
                xhi = min(1.0, frac+width/2)
                arr = generate_grid_rectangular(per_stripe[si], xlo, xhi, 0.06,0.97, jitter_scale=0.012+0.019*random.random())
            else:
                ylo = max(0.0, frac-width/2)
                yhi = min(1.0, frac+width/2)
                arr = generate_grid_rectangular(per_stripe[si], 0.03, 0.97, ylo, yhi, jitter_scale=0.012+0.019*random.random())
            pts.append(arr)
        pts = torch.cat(pts,dim=0)
        pts=pts[torch.randperm(pts.shape[0])][:num_points]
        pts = torch.clamp(pts,0.0,1.0)
        return pts

    def ensure_shape(points):
        if points.shape[0] > problem_size:
            points = points[torch.randperm(points.shape[0])[:problem_size]]
        elif points.shape[0] < problem_size:
            pad = torch.rand(problem_size - points.shape[0], 2)
            points = torch.cat([points, pad], dim=0)
        return points[:problem_size]

    pattern_fns = [
        generate_grid_rectangular,
        generate_band_with_edge_points,
        generate_mild_lattice_patches,
        generate_central_regular_grid_plus_periphery,
        generate_vertical_or_horizontal_stripes
    ]
    pattern_args = [
        lambda ps: (ps, 0.02+0.03*random.random(), 0.98-0.04*random.random(), 0.14+0.03*random.random(), 0.88-0.03*random.random(), 0.02+0.015*random.random()),
        lambda ps: (ps,),
        lambda ps: (ps,),
        lambda ps: (ps,),
        lambda ps: (ps,)
    ]

    out = []
    for i in range(batch_size):
        pat_idx = random.choice(range(len(pattern_fns)))
        problem_fn = pattern_fns[pat_idx]
        args = pattern_args[pat_idx](problem_size)
        pts = problem_fn(*args)
        pts = ensure_shape(pts)
        out.append(pts.float())
    batch = torch.stack(out)
    return batch