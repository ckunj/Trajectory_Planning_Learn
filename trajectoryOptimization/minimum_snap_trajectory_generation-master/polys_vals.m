function vals = polys_vals(polys,ts,tt,r)
    idx = 1;
    N = length(tt);
    vals = zeros(1,N);
    for i = 1:N
        t = tt(i);
        if t<ts(idx)
            vals(i) = 0;
        else
            while idx<length(ts) && t>ts(idx+1)+0.0001
                idx = idx+1;    % 确定是哪一个阶段的多项式
            end
            vals(i) = poly_val(polys(:,idx),t,r);   % 求该阶段多项式的离散点
        end
    end
end

