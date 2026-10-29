def generate_s3_type(batch_size=10, problem_size: int = 100):
    import torch
    import math

    def safe_pad_truncate(points, target_size):
        n = points.size(0)
        if n > target_size:
            perm = torch.randperm(n)
            points = points[perm[:target_size]]
        elif n < target_size:
            pad_count = target_size - n
            pad_pts = torch.rand(pad_count, 2)
            points = torch.cat([points, pad_pts], dim=0)
        return points

    def make_compact_clusters(num_points):
        clusters = torch.randint(3, 7, (1,)).item()
        arr = []
        sizes = [num_points // clusters] * clusters
        for i in range(num_points % clusters):
            sizes[i] += 1
        centroids = torch.rand(clusters, 2) * (0.78 - 0.17) + 0.17
        for i, clsz in enumerate(sizes):
            r_base = 0.035 + torch.rand(1).item() * 0.13
            ov_shape = torch.sqrt(torch.rand(1)).item() if torch.rand(1).item() > 0.58 else 1.
            angle = torch.rand(1).item() * 2 * math.pi
            rotation = torch.tensor([
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle),  math.cos(angle)]
            ])
            cloud = torch.randn(clsz, 2)
            cloud[:, 0] *= r_base * ov_shape
            cloud[:, 1] *= r_base
            clusterpts = torch.matmul(cloud, rotation) + centroids[i]
            arr.append(clusterpts)
        pts = torch.cat(arr, dim=0)
        return safe_pad_truncate(pts, num_points)

    def make_beta_diffuse(num_points):
        a, b = 2.3, 4.8
        q = torch.distributions.Beta(a, b).sample((num_points, 2))
        noise = torch.randn(num_points, 2) * 0.045
        pts = torch.clamp(q + noise, 0.02, 0.98)
        return safe_pad_truncate(pts, num_points)

    def make_grid_diag_noise(num_points):
        side = int(math.ceil(math.sqrt(num_points)) + 1)
        xs = torch.linspace(0.10, 0.90, side)
        ys = torch.linspace(0.10, 0.90, side)
        X, Y = torch.meshgrid(xs, ys)
        grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
        diag_bias = torch.linspace(0,1,grid.size(0)).unsqueeze(1) * 0.017 * (2 * torch.bernoulli(torch.rand(1))-1)
        grid = torch.clamp(grid + diag_bias, 0.0, 1.0)
        idx = torch.randperm(grid.size(0))[:num_points]
        grid = grid[idx]
        jitter = torch.randn(num_points, 2) * (0.024 + torch.rand(1).item() * 0.017)
        return safe_pad_truncate(torch.clamp(grid + jitter, 0.0, 1.0), num_points)

    def make_spiral_circles(num_points):
        circles = torch.randint(3, 6, (1,)).item()
        counts = [num_points // circles] * circles
        for i in range(num_points % circles):
            counts[i] += 1
        allp = []
        for i, s in enumerate(counts):
            spir_ang = i * 0.89 + torch.rand(1).item()
            rad_outer = 0.12 + torch.rand(1).item() * 0.13
            theta = (torch.rand(s) * 2 * math.pi) + spir_ang
            cx = 0.5 + math.cos(2 * math.pi * i / circles + spir_ang) * (0.28 + torch.rand(1).item() * 0.09)
            cy = 0.5 + math.sin(2 * math.pi * i / circles + spir_ang) * (0.25 + torch.rand(1).item() * 0.11)
            local_pts = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
            blobs = (local_pts * rad_outer +
                     torch.tensor([cx, cy], dtype=torch.float32) +
                     torch.randn(s, 2) * (0.017 + torch.rand(1).item() * 0.022))
            allp.append(blobs)
        return safe_pad_truncate(torch.cat(allp, 0), num_points)

    def make_uniform_prism(num_points):
        pts = torch.rand(num_points, 2) * torch.tensor([0.97, 0.88]) + torch.tensor([0.015, 0.06])
        return safe_pad_truncate(pts, num_points)

    def make_hybrid_mixture(num_points):
        clusters = torch.randint(3, 6, (1,)).item()
        s1 = [num_points // clusters] * clusters
        for i in range(num_points % clusters):
            s1[i] += 1
        pts = []
        for i in range(clusters):
            z = torch.rand(1).item()
            if z > 0.69:
                c = torch.rand(2) * (0.85 - 0.15) + 0.15
                r = torch.rand(1).item() * 0.08 + 0.08
                block = c + torch.randn(s1[i], 2) * r
                if torch.rand(1).item() < 0.45:
                    block += torch.randn(1, 2) * 0.017
                pts.append(block)
            elif z > 0.45:
                base = torch.rand(2) * 0.8 + 0.1
                uu = (torch.rand(s1[i], 2)-0.51)*0.17
                pts.append(torch.clamp(base + uu, 0, 1))
            else:
                ang = torch.rand(s1[i]) * 2 * math.pi
                rr = 0.07 + torch.rand(1).item() * 0.13
                center = torch.rand(2) * (0.63 - 0.18) + 0.18
                ring = torch.stack([rr * torch.cos(ang), rr * torch.sin(ang)], dim=1)
                pts.append(center + ring + torch.randn(s1[i], 2) * 0.011)
        pts = torch.cat(pts, 0)
        return safe_pad_truncate(pts, num_points)

    def make_offset_grid_clusters(num_points):
        nrows = torch.randint(2, 4, (1,)).item()
        ncols = nrows
        nclusters = nrows * ncols
        sizes = [num_points // nclusters] * nclusters
        for i in range(num_points % nclusters):
            sizes[i] += 1
        x = torch.linspace(0.17, 0.83, ncols)
        y = torch.linspace(0.2, 0.81, nrows)
        grid_centers = torch.stack([
            x.repeat_interleave(nrows),
            y.repeat(ncols)
        ], dim=1)
        grid_centers += torch.randn_like(grid_centers) * 0.033
        pts = []
        for i in range(nclusters):
            spread = (0.023 + torch.rand(1).item() * 0.04)
            pts.append(grid_centers[i] + torch.randn(sizes[i],2) * spread)
        return safe_pad_truncate(torch.cat(pts,0), num_points)

    def make_field_with_foreground(num_points):
        backprop = 0.64+0.29*torch.rand(1).item()
        n_back = int(num_points * backprop)
        n_back = min(n_back, num_points-2)
        n_back = max(1, n_back)
        n_rem = num_points-n_back
        field = torch.rand(n_back, 2) * torch.tensor([0.88, 0.94]) + torch.tensor([0.06, 0.03])
        n_clusters = torch.randint(2, 4, (1,)).item()
        cl_sizes = [n_rem // n_clusters] * n_clusters
        for i in range(n_rem % n_clusters):
            cl_sizes[i] += 1
        pts = [field]
        for size in cl_sizes:
            c = torch.rand(2)*(0.69-0.14)+0.14
            cr = 0.08 + torch.rand(1).item() * 0.15
            pts.append(c+torch.randn(size,2)*cr)
        return safe_pad_truncate(torch.cat(pts,0), num_points)

    def make_mirrored_clusters(num_points):
        k = torch.randint(2, 6, (1,)).item()
        half = num_points // 2
        sizes = [half // k] * k
        for i in range(half % k):
            sizes[i] += 1
        centers = torch.rand(k, 2) * 0.72 + 0.14
        rad = torch.rand(k) * 0.15 + 0.07
        pts = []
        for idx, sz in enumerate(sizes):
            sz = int(sz)
            bulk = centers[idx] + torch.randn(sz,2)*rad[idx]
            pts.append(bulk)
        main_side = torch.cat(pts,dim=0)
        mirrored = main_side.clone()
        mirrored[:,0] = 1.-mirrored[:,0]
        return safe_pad_truncate(torch.cat([main_side, mirrored],dim=0), num_points)

    def make_annular_blobs(num_points):
        k = torch.randint(4, 8, (1,)).item()
        angles = torch.linspace(0, 2*math.pi, k+1)[:-1] + torch.rand(1).item()*2*math.pi
        ring_rad = torch.rand(1).item()*0.29 + 0.18
        sizes = [num_points // k] * k
        for i in range(num_points % k):
            sizes[i] += 1
        pts = []
        for i, size in enumerate(sizes):
            blobcent = torch.tensor([
                0.5 + math.cos(angles[i].item()) * ring_rad,
                0.5 + math.sin(angles[i].item()) * ring_rad
            ])
            angularity = 1.0 + torch.rand(1).item() * 0.33
            pointss = torch.randn(size, 2)
            pointss[:, 0] *= angularity
            scale_r = 0.072 + torch.rand(1).item() * 0.09
            pts.append(blobcent + pointss * scale_r)
        ap = torch.cat(pts, 0)
        return safe_pad_truncate(ap, num_points)

    def make_uniform_blobs_and_filaments(num_points):
        majority = int(num_points * 0.71)
        fil = num_points - majority
        cat_norm = torch.rand(majority,2) * torch.tensor([0.98,0.94]) + torch.tensor([0.01,0.035])
        ncl = torch.randint(2, 4, (1,)).item()
        asg = [fil // ncl] * ncl
        for i in range(fil % ncl):
            asg[i] += 1
        blobs = []
        for blockn in asg:
            c = torch.rand(2) * 0.67 + 0.17
            direction = torch.tensor([
                math.cos(torch.rand(1).item()*2*math.pi),
                math.sin(torch.rand(1).item()*2*math.pi)])
            spread = torch.rand(1).item()*0.075+0.017
            blockpts = c+torch.randn(blockn,2)*spread
            if torch.rand(1).item() < 0.58 and blockn >= 5:
                parts = torch.linspace(0., 1., blockn).unsqueeze(1)
                follower = (parts-0.5)*torch.rand(1).item()*0.14*direction
                blockpts = c+follower + torch.randn(blockn,2) * spread * 0.55
            blobs.append(blockpts)
        blobs.insert(0, cat_norm)
        result = torch.cat(blobs,0)
        return safe_pad_truncate(result, num_points)

    GENS = [
        make_compact_clusters,
        make_beta_diffuse,
        make_grid_diag_noise,
        make_spiral_circles,
        make_uniform_prism,
        make_hybrid_mixture,
        make_offset_grid_clusters,
        make_field_with_foreground,
        make_mirrored_clusters,
        make_annular_blobs,
        make_uniform_blobs_and_filaments
    ]

    batch = []
    for _ in range(batch_size):
        sel = torch.randint(0, len(GENS), (1,)).item()
        points = GENS[sel](problem_size)
        if torch.rand(1).item() > 0.81:
            points = points + torch.randn_like(points) * (0.018 + torch.rand(1).item() * 0.018)
        if torch.rand(1).item() > 0.93:
            points = points * (0.91 + torch.rand(1).item() * 0.07) + torch.rand(1).item()*0.048
        points = torch.clamp(points, 0.0, 1.0)
        points = torch.nan_to_num(points, nan=0.5, posinf=1.0, neginf=0.0)
        points = points.reshape(problem_size, 2).to(torch.float32)
        batch.append(points)
    return torch.stack(batch, 0)

