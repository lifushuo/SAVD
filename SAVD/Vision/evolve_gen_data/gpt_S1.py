def generate_s1_type(batch_size, problem_size: int = 100):
    import torch
    import random
    import math

    def grid_block(origin, span, num_x, num_y, jitter=0.0015):
        x = torch.linspace(0, span[0], max(1, num_x))
        y = torch.linspace(0, span[1], max(1, num_y))
        xx, yy = torch.meshgrid(x, y)
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        grid += (torch.rand_like(grid) - 0.5) * jitter * 0.90
        return origin + grid

    def band_arc(center, r, total_pts, angle_from, angle_to, drift=0.0052, thickness=0.036):
        phi = torch.linspace(angle_from, angle_to, max(1, total_pts))
        x = torch.cos(phi) * r + center[0]
        y = torch.sin(phi) * r + center[1]
        arc = torch.stack([x, y], dim=-1)
        arc += (torch.rand_like(arc) - 0.5) * thickness * 0.96
        arc += (torch.rand_like(arc) - 0.5) * drift * 0.14
        return arc

    def l_shaped_block(origin, main_len, arm_len, pts_per_arm, thickness=0.0041):
        hor = random.random() < 0.43
        count = max(2, pts_per_arm)
        if hor:
            p1 = torch.cat([torch.linspace(0, main_len, count).unsqueeze(1), torch.zeros(count, 1)], dim=1)
            p2 = torch.cat([torch.full((count, 1), main_len), torch.linspace(0, arm_len, count).unsqueeze(1)], dim=1)
        else:
            p1 = torch.cat([torch.zeros(count, 1), torch.linspace(0, main_len, count).unsqueeze(1)], dim=1)
            p2 = torch.cat([torch.linspace(0, arm_len, count).unsqueeze(1), torch.full((count, 1), main_len)], dim=1)
        l = torch.cat([p1, p2], dim=0)
        l += (torch.rand_like(l) - 0.5) * thickness * 1.36
        l += origin
        return l

    def block_cross(center, side, n_per_arm, jitter=0.008):
        count = max(2, n_per_arm)
        h = torch.cat([torch.linspace(-side/2, side/2, count).unsqueeze(1), torch.zeros(count,1)], dim=1)
        v = torch.cat([torch.zeros(count,1), torch.linspace(-side/2, side/2, count).unsqueeze(1)], dim=1)
        pts = torch.cat([h, v], dim=0)
        pts += (torch.rand_like(pts)-0.5) * jitter * 0.91
        return center + pts

    def scattered_blob(center, scale, n):
        pts = (torch.rand(max(1, n), 2) - 0.5) * scale * 0.73 + center
        return pts

    def global_affine_norm(points):
        mins = torch.min(points, dim=0, keepdim=True)[0]
        maxs = torch.max(points, dim=0, keepdim=True)[0]
        span = maxs - mins
        span = torch.clamp(span, min=1e-4)
        normed = (points - mins) / span
        theta = math.pi * (0.43 + random.random()*0.09)
        rot = torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=torch.float32)
        normed = (normed - 0.5) @ rot + 0.5
        if random.random() < 0.34:
            normed[:, 1] = 1 - normed[:, 1]
        normed += (torch.rand_like(normed) - 0.5) * 0.0011
        return torch.clamp(normed, 0.0, 1.0)

    def cluster_grid_block(centroid, n_pts, pattern_type):
        if pattern_type == 'square':
            side = max(1, int(round(n_pts ** 0.5)))
            x = torch.linspace(-0.038, 0.038, side)
            y = torch.linspace(-0.038, 0.038, side)
            grid = torch.stack(torch.meshgrid(x, y), dim=-1).reshape(-1,2)
            motif = grid[:n_pts]
        elif pattern_type == 'rect':
            rows = max(1, random.choice([3, 4]))
            cols = max(1, (n_pts + rows - 1) // rows)
            x = torch.linspace(-0.034, 0.038, cols)
            y = torch.linspace(-0.026, 0.034, rows)
            grid = torch.stack(torch.meshgrid(x, y), dim=-1).reshape(-1,2)
            motif = grid[:n_pts]
        else:
            motif = (torch.rand(n_pts,2) - 0.5) * 0.054
        motif += centroid
        return motif

    def clustered_problem(problem_size):
        pattern_type = random.choices(['square', 'rect', 'scatter'], weights=[0.52, 0.28, 0.20])[0]
        num_centers = random.choice([2, 4, 6])
        cluster_centers = (torch.rand(num_centers,2) * 0.43 + 0.24)
        pts_per_cluster = max(1, (problem_size + num_centers - 1) // num_centers)
        motifs = [cluster_grid_block(center, pts_per_cluster, pattern_type) for center in cluster_centers]
        points = torch.cat(motifs, dim=0)
        if points.size(0) > problem_size:
            idx = torch.randperm(points.size(0))[:problem_size]
            points = points[idx]
        elif points.size(0) < problem_size:
            to_pad = problem_size - points.size(0)
            points = torch.cat([points, torch.rand(to_pad, 2)], dim=0)
        points = points.float()
        points = global_affine_norm(points)
        return points

    def band_motif(center, angle_rad, length, width, n_pts):
        t = torch.rand(n_pts, 1)
        pos = (t - 0.5) * length
        offset = (torch.rand(n_pts,1) - 0.5) * width
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        axis = torch.tensor([dx, dy], dtype=torch.float32).reshape(1,2)
        perp = torch.tensor([-dy, dx], dtype=torch.float32).reshape(1,2)
        pts = center.reshape(1,2) + axis*pos + perp*offset
        return pts

    def grid_motif(center, rows, cols, span, jitter, n_pts):
        x = torch.linspace(-span / 2, span / 2, max(1, cols))
        y = torch.linspace(-span / 2, span / 2, max(1, rows))
        grid = torch.stack(torch.meshgrid(x, y), dim=-1).reshape(-1,2)
        idx = torch.randperm(grid.size(0))
        grid = grid[idx[:n_pts]]
        pts = grid + center.reshape(1,2)
        pts = pts + (torch.rand_like(pts)-0.5)*jitter
        return pts

    def radial_star(center, spokes, pts_per_spoke, radius, jitter):
        all_pts = []
        angle_offset = random.uniform(0, 2*math.pi)
        for s in range(spokes):
            angle = angle_offset + 2 * math.pi * s / spokes
            r = torch.linspace(radius*0.18, radius, max(1, pts_per_spoke))
            xs = r * math.cos(angle)
            ys = r * math.sin(angle)
            pts = torch.stack([xs, ys], dim=1)
            pts += (torch.rand_like(pts)-0.5) * jitter * 0.89
            pts += center.reshape(1,2)
            all_pts.append(pts)
        pts = torch.cat(all_pts, dim=0)
        return pts

    def block_lattice(centers, rows, cols, span, jitter, per_block):
        blocks = []
        for center in centers:
            blocks.append(grid_motif(center, rows, cols, span, jitter, per_block))
        return torch.cat(blocks, dim=0)

    def ellipse_frame(center, a, b, n_pts, jitter=0.0012):
        theta = torch.linspace(0, 2 * math.pi, n_pts + 1)[:-1]
        x = a * torch.cos(theta)
        y = b * torch.sin(theta)
        pts = torch.stack([x, y], dim=1) + center.reshape(1,2)
        pts = pts + (torch.rand_like(pts) - 0.5) * jitter * 1.03
        return pts

    def scatter_blob(center, n_pts, scale):
        blob = (torch.rand(n_pts, 2) - 0.5) * scale
        return center.reshape(1,2) + blob

    def motif_mixture(problem_size):
        motif_types = ['band', 'cross', 'grid', 'lblock', 'block', 'scatter', 'star', 'ellipse']
        chosen_k = 3 + random.randint(0,2)
        roots = torch.linspace(0, 2*math.pi, chosen_k+1)[:-1]
        random.shuffle(motif_types)
        chosen_types = motif_types[:chosen_k]
        splits_raw = torch.rand(chosen_k)
        splits = splits_raw / splits_raw.sum()
        min_pts_motif = max(4, int(problem_size*0.049))
        pts_per_motif = (splits * (problem_size - chosen_k * min_pts_motif)).long() + min_pts_motif
        correction = int(problem_size - pts_per_motif.sum().item())
        pts_per_motif[0] += correction
        ring_r = 0.295 + 0.07 * random.random()
        jitter_ring = torch.rand(chosen_k) * 0.021

        centerpoints = [
            torch.tensor([
                0.50 + (math.cos(roots[i]) * (ring_r + jitter_ring[i]).item()),
                0.50 + (math.sin(roots[i]) * (ring_r + jitter_ring[i]).item())
            ], dtype=torch.float32)
            for i in range(chosen_k)
        ]
        motif_points = []
        for idx, name in enumerate(chosen_types):
            local_n = int(pts_per_motif[idx].item())
            center = centerpoints[idx]
            if name == 'band':
                angle0 = random.uniform(-math.pi, math.pi)
                sweep = random.uniform(0.41, 1.11)
                r = random.uniform(0.11, 0.18)
                band = band_arc(center, r, local_n, angle0, angle0 + sweep,
                                drift=random.uniform(0.003,0.007),
                                thickness=random.uniform(0.020,0.034))
                motif_points.append(band)
            elif name == 'cross':
                arm_len = random.uniform(0.064, 0.085)
                crs = block_cross(center, arm_len, (local_n+1)//2, jitter=random.uniform(0.0032,0.006))
                motif_points.append(crs[:local_n])
            elif name == 'grid':
                grid_nx = max(2, int(round(math.sqrt(local_n))))
                grid_ny = max(2, (local_n+grid_nx-1)//grid_nx)
                span = (random.uniform(0.051,0.071), random.uniform(0.059,0.085))
                base = grid_block(center - torch.tensor(span)/2, span, grid_nx, grid_ny, jitter=random.uniform(0.0022,0.0035))
                motif_points.append(base[:local_n])
            elif name == 'block':
                grid_nx = max(2, int(math.floor(math.sqrt(local_n))))
                grid_ny = max(2, local_n//grid_nx)
                span = (random.uniform(0.057,0.077), random.uniform(0.065,0.089))
                base = grid_block(center-torch.tensor(span)/2, span, grid_nx, grid_ny, jitter=random.uniform(0.0018,0.0028))
                motif_points.append(base[:local_n])
            elif name == 'lblock':
                mainl = random.uniform(0.051,0.067)
                arml = random.uniform(0.038,0.059)
                arm_n = max(2, (local_n)//2)
                motif = l_shaped_block(center - torch.tensor([mainl/2, arml/2]), mainl, arml, arm_n, thickness=random.uniform(0.0016,0.0039))
                motif_points.append(motif[:local_n])
            elif name == 'scatter':
                scale = random.uniform(0.047,0.058)
                motif_points.append(scattered_blob(center, scale, local_n))
            elif name == 'star':
                sp = random.randint(3, 4)
                pts_per_spoke = max(2, (local_n + sp - 1) // sp)
                rad = random.uniform(0.061, 0.093)
                jitter_val = random.uniform(0.0037, 0.008)
                out = radial_star(center, sp, pts_per_spoke, rad, jitter_val)
                if out.shape[0] > local_n:
                    out = out[torch.randperm(out.shape[0])[:local_n]]
                elif out.shape[0] < local_n:
                    padnum = local_n - out.shape[0]
                    out = torch.cat([out, scatter_blob(center, padnum, 0.0095)], dim=0)
                motif_points.append(out)
            elif name == 'ellipse':
                a = random.uniform(0.053,0.074)
                b = random.uniform(0.036,0.059)
                n_pts = local_n
                out = ellipse_frame(center, a, b, n_pts)
                motif_points.append(out)
        pts = torch.cat(motif_points, dim=0)
        if pts.shape[0] > problem_size:
            idx = torch.randperm(pts.shape[0])[:problem_size]
            pts = pts[idx]
        elif pts.shape[0] < problem_size:
            to_pad = problem_size - pts.shape[0]
            pad = torch.rand(to_pad,2)
            pts = torch.cat([pts, pad], dim=0)
        pts = pts.float()
        pts = global_affine_norm(pts)
        return pts

    batch_lst = []
    for _ in range(batch_size):
        ptype = random.random()
        if ptype < 0.48:
            pts = clustered_problem(problem_size)
        else:
            pts = motif_mixture(problem_size)
        if pts.shape[0] > problem_size:
            idx = torch.randperm(pts.shape[0])[:problem_size]
            pts = pts[idx]
        elif pts.shape[0] < problem_size:
            excess = problem_size - pts.shape[0]
            pts = torch.cat([pts, torch.rand(excess,2)], dim=0)
        pts = pts.reshape(problem_size,2)
        batch_lst.append(pts)
    batch = torch.stack(batch_lst, dim=0).float()
    return batch